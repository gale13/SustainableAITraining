// Exact compiled evaluation of the frozen full-data MLP and empirical bank.
// No fitting, interpolation of the model, or alternate loss dynamics occurs.
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <limits>
static double jw_mean,jw_sd,initial_J,beta[2],weightsJ[3][16],intercepts[3][3][5][16],outputW[3][16],bias[3],scale[3];
static int widths[3];
static double tr[5][4][30],dr[5][4][30],vr[5][4][30],vshift[5][4];
static std::vector<double> knots,pv,prefix;
static std::vector<double> observation_times,observation_power;
extern "C" int choose_action(const double*,double,double,double,double,double,double*,double*);
static bool invalid_window=false;
static const int ACTIONS[5]={32,64,128,256,512};
static const int BASE[6][4]={{0,0,0,0},{2,2,2,2},{4,4,4,4},{0,0,4,4},{0,1,3,4},{4,3,1,0}};

extern "C" void configure(const double* spec,const double* W,const double* B,const double* V,
                           const double* metadata,const double* bank,const double* solar,int nsolar){
    jw_mean=spec[0];jw_sd=spec[1];initial_J=spec[2];beta[0]=spec[3];beta[1]=spec[4];
    for(int m=0;m<3;m++){
        widths[m]=(int)metadata[m*3];bias[m]=metadata[m*3+1];scale[m]=metadata[m*3+2];
        for(int h=0;h<widths[m];h++){
            weightsJ[m][h]=W[m*9*16+h];outputW[m][h]=V[m*16+h];
            for(int p=0;p<3;p++)for(int a=0;a<5;a++)
                intercepts[m][p][a][h]=B[m*16+h]+W[m*9*16+(1+p)*16+h]+W[m*9*16+(4+a)*16+h];
        }
    }
    for(int a=0;a<5;a++)for(int e=0;e<4;e++){
        double mt=0,md=0;
        for(int i=0;i<30;i++){int k=((i*5+a)*4+e)*3;mt+=bank[k]/30.;md+=bank[k+1]/30.;}
        vshift[a][e]=0;
        for(int i=0;i<30;i++){
            int k=((i*5+a)*4+e)*3;
            tr[a][e][i]=bank[k]/mt;dr[a][e][i]=bank[k+1]/md;vr[a][e][i]=bank[k+2];
            vshift[a][e]-=tr[a][e][i]*vr[a][e][i]/30.;
        }
    }
    // PV is supplied separately through the audited native-knot adapter.
    (void)solar; (void)nsolar;
}

static void means(double J,int phase,int action,double* out){
    double x=(J-jw_mean)/jw_sd;
    for(int m=0;m<3;m++){
        double z=bias[m];
        for(int h=0;h<widths[m];h++)z+=std::tanh(weightsJ[m][h]*x+intercepts[m][phase-1][action][h])*outputW[m][h];
        if(m<2)out[m]=scale[m]*(std::max(z,0.)+std::log1p(std::exp(-std::abs(z))));
        else out[m]=out[0]*J/(1.+std::exp(-std::clamp(z,-50.,50.)));
    }
}
extern "C" void predict_means(double J,int phase,int action,double* out){ means(J,phase,action,out); }
extern "C" void set_observations(const double* times,const double* power,int n){
    observation_times.assign(times,times+n);observation_power.assign(power,power+n);
}
extern "C" void latest_observation(double time,double multiplier,double* out){
    auto found=std::upper_bound(observation_times.begin(),observation_times.end(),time);
    if(!std::isfinite(time)||time<0||time>86400||found==observation_times.begin()){
        invalid_window=true;out[0]=out[1]=out[2]=std::numeric_limits<double>::quiet_NaN();return;
    }
    int i=(int)(found-observation_times.begin())-1;
    out[0]=multiplier*observation_power[i];out[1]=observation_times[i];out[2]=time-out[1];
}
extern "C" void policy_decision(double J,int phase,double Q,double time,double V,
                                  double weight,double sigma,double multiplier,double* out){
    double observed[3],m[5][3],shortfall[5],scores[5];
    latest_observation(time,multiplier,observed);
    for(int a=0;a<5;a++)means(J,phase,a,m[a]);
    int chosen=choose_action(m[0],V,weight,Q,sigma,observed[0],shortfall,scores);
    out[0]=chosen;std::copy(observed,observed+3,out+1);
    for(int a=0;a<5;a++){
        out[4+a*6]=m[a][0];out[5+a*6]=m[a][1];out[6+a*6]=m[a][2];
        out[7+a*6]=observed[0]*m[a][0];out[8+a*6]=shortfall[a];out[9+a*6]=scores[a];
    }
}
// Exact nonperiodic piecewise-linear energy. Knots come through PVDatabase.
extern "C" void set_pv(const double* times,const double* power,int n){
    knots.assign(times,times+n);pv.assign(power,power+n);prefix.assign(n,0.);
    invalid_window=false;
    for(int i=1;i<n;i++)prefix[i]=prefix[i-1]+.5*(pv[i-1]+pv[i])*(knots[i]-knots[i-1]);
}
static double integral_to(double seconds){
    if(knots.size()<2 || seconds<knots.front() || seconds>knots.back()){
        invalid_window=true;return std::numeric_limits<double>::quiet_NaN();
    }
    if(seconds==knots.back())return prefix.back();
    int i=(int)(std::upper_bound(knots.begin(),knots.end(),seconds)-knots.begin())-1;
    double dt=seconds-knots[i],slope=(pv[i+1]-pv[i])/(knots[i+1]-knots[i]);
    return prefix[i]+pv[i]*dt+.5*slope*dt*dt;
}
static double renewable(double time,double duration,double multiplier){
    double value=multiplier*(integral_to(time+duration)-integral_to(time));
    if(!std::isfinite(value))return value;
    return std::max(0.,value);
}
extern "C" double pv_energy(double time,double duration,double multiplier){
    return renewable(time,duration,multiplier);
}
extern "C" int pv_invalid(){return invalid_window?1:0;}
static double retention_shift(double q,int a,int e){
    double shift=vshift[a][e];bool interior=true;
    for(int i=0;i<30;i++){double value=q+vr[a][e][i]+shift;if(value<1e-12||value>1.){interior=false;break;}}
    if(interior)return shift;
    double low=1e100,high=-1e100;
    for(int i=0;i<30;i++){low=std::min(low,q+vr[a][e][i]);high=std::max(high,q+vr[a][e][i]);}
    double lo=-high,hi=1.-low;
    for(int iteration=0;iteration<60;iteration++){
        double mid=(lo+hi)/2.,mean=0.;
        for(int i=0;i<30;i++)mean+=tr[a][e][i]*std::clamp(q+vr[a][e][i]+mid,1e-12,1.)/30.;
        if(mean<q)lo=mid;else hi=mid;
    }
    return (lo+hi)/2.;
}
static int next_phase(int phase,double J,double jp){
    double delta=(jp-J)/(J+1e-8);int candidate=delta<=beta[0]?1:(delta<=beta[1]?2:3);
    return std::max(phase,candidate);
}
extern "C" void inspect(double J,int phase,int a,int e,double time,double multiplier,double* out){
    double m[3];means(J,phase,a,m);std::copy(m,m+3,out);
    double observed[3];latest_observation(time,multiplier,observed);
    out[3]=std::max(0.,m[1]-observed[0]*m[0]);
    double q=m[2]/(m[0]*J),shift=retention_shift(q,a,e);
    for(int i=0;i<30;i++){
        double t=m[0]*tr[a][e][i],d=m[1]*dr[a][e][i],jp=J*std::clamp(q+vr[a][e][i]+shift,1e-12,1.);
        out[4+i*4]=t;out[5+i*4]=d;out[6+i*4]=t*jp;out[7+i*4]=jp;
    }
}
// Summary: T,D,PV,shortfall,epsilon,sumQ,Qend,maxQ,sum_terminal_J,
// five action counts, three phase counts, and task count.
// Trace columns are specified by TRACE_COLUMNS in runtime.py.
extern "C" void run(int tasks,int policy,double V,const double* phaseweights,double sigma,
                     double multiplier,double starttime,const unsigned char* donors,double* summary,double* trace){
    invalid_window=false;
    std::fill(summary,summary+24,0.);double time=starttime,Q=0.,J=initial_J;int phase=1;
    for(int epoch=0;epoch<tasks*4;epoch++){
        int e=epoch%4;if(e==0){J=initial_J;phase=1;}
        int chosen=policy==0?0:BASE[policy-1][e];double m[5][3],estimated_es[5]={0},scores[5],theta=0.;
        double weight=phaseweights[phase-1];
        double observed[3];latest_observation(time,multiplier,observed);
        if(policy==0){
            for(int a=0;a<5;a++)means(J,phase,a,m[a]);
            chosen=choose_action(m[0],V,weight,Q,sigma,observed[0],estimated_es,scores);
            theta=scores[chosen];
        }else{
            means(J,phase,chosen,m[chosen]);
            estimated_es[chosen]=std::max(0.,m[chosen][1]-observed[0]*m[chosen][0]);
        }
        int donor=(int)donors[epoch];double mt=m[chosen][0],md=m[chosen][1],mh=m[chosen][2];
        double q=mh/(mt*J),shift=retention_shift(q,chosen,e);
        double t=mt*tr[chosen][e][donor],d=md*dr[chosen][e][donor];
        double jp=J*std::clamp(q+vr[chosen][e][donor]+shift,1e-12,1.);
        double ren=renewable(time,t,multiplier),es=std::max(0.,d-ren),eps=t*jp;
        double nextQ=std::max(0.,Q+eps-sigma*t);int np=next_phase(phase,J,jp);
        if(trace){
            double row[28]={(double)(epoch+1),(double)(epoch/4+1),(double)(e+1),time,(double)phase,(double)np,
                (double)ACTIONS[chosen],(double)(donor+1),J,jp,t,d,ren,es,eps,Q,nextQ,weight,mt,md,mh,estimated_es[chosen],theta,
                (double)(policy==0),observed[0],observed[1],observed[2],observed[0]*mt};
            std::copy(row,row+28,trace+epoch*28);
        }
        summary[0]+=t;summary[1]+=d;summary[2]+=ren;summary[3]+=es;summary[4]+=eps;summary[5]+=Q;
        summary[6]=nextQ;summary[7]=std::max(summary[7],nextQ);if(e==3)summary[8]+=jp;
        summary[9+chosen]++;summary[14+phase-1]++;summary[17]=tasks;
        summary[18]+=observed[2];summary[19]=std::max(summary[19],observed[2]);
        summary[20]+=(observed[2]>300);summary[21]+=(observed[2]>600);
        summary[22]+=observed[0];summary[23]+=observed[0]*mt;
        J=jp;phase=np;Q=nextQ;time+=t;
    }
}
