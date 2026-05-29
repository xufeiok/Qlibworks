import json

json_path = r'e:\Quant\Qlibworks\extracted_gtja191.json'
with open(json_path, 'r', encoding='utf-8') as f:
    formulas = json.load(f)

overrides = {
    3: "Sum(If($close == Ref($close,1), 0, If($close > Ref($close,1), $close - Less($low, Ref($close,1)), $close - Greater($high, Ref($close,1)))), 6)",
    4: "If((((Sum($close,8)/8)+Std($close,8))<(Sum($close,2)/2)), (-1*1), If(((Sum($close,2)/2)<((Sum($close,8)/8)-Std($close,8))), 1, If(((1<($volume/Mean($volume,20))) | (($volume/Mean($volume,20))==1)), 1, (-1*1))))",
    10: "(Rank(Max(If((($close/Ref($close,1)-1)<0), Std(($close/Ref($close,1)-1),20), $close)^2),5))",
    19: "If($close<Ref($close,5), ($close-Ref($close,5))/Ref($close,5), If($close==Ref($close,5), 0, ($close-Ref($close,5))/$close))",
    23: "Sma(If($close>Ref($close,1), Std($close, 20), 0),20,1)/(Sma(If($close>Ref($close,1), Std($close,20), 0),20,1)+Sma(If($close<=Ref($close,1), Std($close,20), 0),20,1))*100",
    38: "If(((Sum($high,20)/20)<$high), (-1*Delta($high,2)), 0)",
    40: "Sum(If($close>Ref($close,1), $volume, 0),26)/Sum(If($close<=Ref($close,1), $volume, 0),26)*100",
    43: "Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)),6)",
    49: "Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12))",
    50: "Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12))-Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12))",
    51: "Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Max(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12))",
    55: "Sum(16*($close-Ref($close,1)+($close-$open)/2+Ref($close,1)-Ref($open,1))/(If((Abs($high-Ref($close,1))>Abs($low-Ref($close,1))) & (Abs($high-Ref($close,1))>Abs($high-Ref($low,1))), Abs($high-Ref($close,1))+Abs($low-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, If((Abs($low-Ref($close,1))>Abs($high-Ref($low,1))) & (Abs($low-Ref($close,1))>Abs($high-Ref($close,1))), Abs($low-Ref($close,1))+Abs($high-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, Abs($high-Ref($low,1))+Abs(Ref($close,1)-Ref($open,1))/4)))*Max(Abs($high-Ref($close,1)),Abs($low-Ref($close,1))),20)",
    59: "Sum(If($close==Ref($close,1), 0, $close-If($close>Ref($close,1), Min($low,Ref($close,1)), Max($high,Ref($close,1)))),20)",
    69: "If(Sum(DTM,20)>Sum(DBM,20), (Sum(DTM,20)-Sum(DBM,20))/Sum(DTM,20), If(Sum(DTM,20)==Sum(DBM,20), 0, (Sum(DTM,20)-Sum(DBM,20))/Sum(DBM,20)))",
    84: "Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)),20)",
    86: "If((0.25<(((Ref($close,20)-Ref($close,10))/10)-((Ref($close,10)-$close)/10))), (-1*1), If(((((Ref($close,20)-Ref($close,10))/10)-((Ref($close,10)-$close)/10))<0), 1, ((-1*1)*($close-Ref($close,1)))))",
    93: "Sum(If($open>=Ref($open,1), 0, Max(($open-$low),($open-Ref($open,1)))),20)",
    94: "Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)),30)",
    98: "If((((Delta((Sum($close,100)/100),100)/Ref($close,100))<0.05) | ((Delta((Sum($close,100)/100),100)/Ref($close,100))==0.05)), (-1*($close-Ts_Min($close,100))), (-1*Delta($close,3)))",
    112: "(Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)-Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12))/(Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)+Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12))*100",
    128: "100-(100/(1+Sum(If(($high+$low+$close)/3>Ref(($high+$low+$close)/3,1), ($high+$low+$close)/3*$volume, 0),14)/Sum(If(($high+$low+$close)/3<Ref(($high+$low+$close)/3,1), ($high+$low+$close)/3*$volume, 0),14)))",
    129: "Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12)",
    137: "16*($close-Ref($close,1)+($close-$open)/2+Ref($close,1)-Ref($open,1))/(If((Abs($high-Ref($close,1))>Abs($low-Ref($close,1))) & (Abs($high-Ref($close,1))>Abs($high-Ref($low,1))), Abs($high-Ref($close,1))+Abs($low-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, If((Abs($low-Ref($close,1))>Abs($high-Ref($low,1))) & (Abs($low-Ref($close,1))>Abs($high-Ref($close,1))), Abs($low-Ref($close,1))+Abs($high-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, Abs($high-Ref($low,1))+Abs(Ref($close,1)-Ref($open,1))/4)))*Max(Abs($high-Ref($close,1)),Abs($low-Ref($close,1)))",
    143: "If($close>Ref($close,1), ($close-Ref($close,1))/Ref($close,1)*SELF, SELF)",
    160: "Sma(If($close<=Ref($close,1), Std($close,20), 0),20,1)",
    164: "Sma((If($close>Ref($close,1), 1/($close-Ref($close,1)), 1)-Min(If($close>Ref($close,1), 1/($close-Ref($close,1)), 1),12))/($high-$low)*100,13,2)",
    167: "Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)",
    172: "Mean(Abs(Sum(If(LD>0 & LD>HD, LD, 0),14)*100/Sum(TR,14)-Sum(If(HD>0 & HD>LD, HD, 0),14)*100/Sum(TR,14))/(Sum(If(LD>0 & LD>HD, LD, 0),14)*100/Sum(TR,14)+Sum(If(HD>0 & HD>LD, HD, 0),14)*100/Sum(TR,14))*100,6)",
    174: "Sma(If($close>Ref($close,1), Std($close,20), 0),20,1)",
    180: "If(Mean($volume,20)<$volume, (-1*Ts_Rank(Abs(Delta($close,7)),60))*Sign(Delta($close,7)), -1*$volume)",
    186: "(Mean(Abs(Sum(If(LD>0 & LD>HD, LD, 0),14)*100/Sum(TR,14)-Sum(If(HD>0 & HD>LD, HD, 0),14)*100/Sum(TR,14))/(Sum(If(LD>0 & LD>HD, LD, 0),14)*100/Sum(TR,14)+Sum(If(HD>0 & HD>LD, HD, 0),14)*100/Sum(TR,14))*100,6)+Ref(Mean(Abs(Sum(If(LD>0 & LD>HD, LD, 0),14)*100/Sum(TR,14)-Sum(If(HD>0 & HD>LD, HD, 0),14)*100/Sum(TR,14))/(Sum(If(LD>0 & LD>HD, LD, 0),14)*100/Sum(TR,14)+Sum(If(HD>0 & HD>LD, HD, 0),14)*100/Sum(TR,14))*100,6),6))/2",
    187: "Sum(If($open<=Ref($open,1), 0, Max(($high-$open),($open-Ref($open,1)))),20)",
    8: "Rank(Delta((((($high+$low)/2)*0.2)+($vwap*0.8)),4))-1",
    9: "Sma((($high+$low)/2-(Ref($high,1)+Ref($low,1)))/($high-$low)/$volume,7,2)",
    11: "(Sum((($close-$low)-($high-$close))/($high-$low)*$volume,6))",
    31: "($close-Mean($close,12))/Mean($close,12)*100",
    39: "((Rank(Decaylinear(Delta(($close),2),8))-Rank(Decaylinear(Corr((($vwap*0.3)+($open*0.7)),Sum(Mean($volume,180),37),14),12)))*-1)",
    42: "((-1*Rank(Std($high,10)))*Corr($high,$volume,10))",
    75: "Count($close<$open,50)/Count($close<$open,50)",
    101: "((Rank(Corr($close,Sum(Mean($volume,30),37),15))))",
    123: "((Rank(Corr(Sum((($high+$low)/2),20),Sum(Mean($volume,60),20),9))))"
}

print(f"Total overrides: {len(overrides)}")
import json
with open(r'e:\Quant\Qlibworks\overrides.json', 'w') as f:
    json.dump(overrides, f, indent=4)
