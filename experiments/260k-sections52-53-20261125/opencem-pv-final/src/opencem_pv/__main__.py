import argparse,json
from pathlib import Path
from .api import PVDatabase

def main():
    parser=argparse.ArgumentParser(description='OpenCEM inverter-1 daily PV data')
    parser.add_argument('--database',type=Path)
    sub=parser.add_subparsers(dest='command',required=True)
    listing=sub.add_parser('days');listing.add_argument('--recommended',action='store_true')
    day=sub.add_parser('day');day.add_argument('date');day.add_argument('--step',type=float,default=1)
    day.add_argument('--scale',type=float,default=1);day.add_argument('--max-gap',type=float,default=300)
    day.add_argument('--output',type=Path,required=True)
    energy=sub.add_parser('energy');energy.add_argument('date');energy.add_argument('--start',type=float,required=True)
    energy.add_argument('--duration',type=float,required=True);energy.add_argument('--scale',type=float,default=1)
    args=parser.parse_args()
    with PVDatabase(args.database) as db:
        if args.command=='days':print(db.days(recommended_only=args.recommended).to_csv(index=False),end='')
        elif args.command=='day':
            frame=db.load_day(args.date,max_gap_s=args.max_gap,scale=args.scale).sample(args.step)
            args.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output,index=False)
            print(json.dumps(dict(output=str(args.output),rows=len(frame),missing=int(frame.pv_power_w.isna().sum()))))
        else:
            value=db.load_day(args.date,scale=args.scale).energy(args.start,args.duration)
            print(json.dumps(dict(energy_j=value,energy_wh=value/3600)))

if __name__=='__main__':main()
