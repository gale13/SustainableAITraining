// Pure decision function. Its inputs contain no noise bank, donor IDs, clock,
// future PV trajectory, or environmental scenario interface.
#include <algorithm>
#include <limits>

extern "C" int choose_action(const double* means, double V, double weight,
                              double Q, double sigma, double observed_power,
                              double* estimated_shortfall, double* scores) {
    int chosen=0;
    double best=std::numeric_limits<double>::infinity();
    for(int a=0;a<5;a++) {
        const double T=means[3*a], D=means[3*a+1], epsilon=means[3*a+2];
        estimated_shortfall[a]=std::max(0.,D-observed_power*T);
        scores[a]=(V*estimated_shortfall[a]+weight*Q*(epsilon-sigma*T))/T;
        if(scores[a]<best) { best=scores[a]; chosen=a; }
    }
    return chosen;
}
