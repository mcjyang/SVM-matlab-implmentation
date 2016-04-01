clc; clear all; close all;

% use SGD for SVM training

% Question 4.1
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
C=0.1;
[w,b]=SVM_SGD(trD',trLb,C);
HW2_Utils.genRsltFile(w, b, 'val', './data/result_val');
[ap, prec, rec]=HW2_Utils.cmpAP('./data/result_val','val');


% Question 4.2

C=0.05;
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
[w,b,obj,index]=SVM_QP(trD',trLb,C);

n=size(trLb,1);
PostrD=trD(:,1:176);
NegtrD=trD(:,177:n);
PosLb=trLb(1:176);
NegLb=trLb(177:n);

sv=index(find(index>176));
disp('initial state:');
disp(['objective value: ',num2str(obj)]);
disp(['total data:',num2str(size(trD,2))]);
disp(['total svs:',num2str(size(index,1))]);
disp(['negative svs:',num2str(size(sv,1))]);
% y=valD'*w+b;
% result=find(y.*valLb>0);
% disp(['accuracy: ',num2str(size(result,1)/size(valLb,1))]);

HW2_Utils.genRsltFile(w, b, 'val', './data/result_val');
[ap, prec, rec]=HW2_Utils.cmpAP('./data/result_val','val');
close();
disp(['ap:',num2str(ap)]);




% Hard negative mining algorithm

load(sprintf('%s/%sAnno.mat', HW2_Utils.dataDir, 'train'), 'ubAnno');

Data=trD;
Lb=trLb;
num=100;

ws=[w];
bs=[b];
objs=[obj];
aps=[ap];

for k=1:10
      
      NegLb=Lb(sv);
      NegtrD=Data(:,sv');

      [feature,label,rects] = findExample(num,w,b,k,ubAnno);
      
      NegLb=cat(1,NegLb,label);
      Lb=cat(1,PosLb,NegLb);
      NegtrD=cat(2,NegtrD,feature);
      Data=cat(2,PostrD,NegtrD);
      
      [w,b,obj,index]=SVM_QP(double(Data'),Lb,C);
      sv=index(find(index>176));
      disp(['iteration: ',num2str(k)]);
      disp(['objective value: ',num2str(obj)]);
      disp(['total data:',num2str(size(Data,2))]);
      disp(['total svs:',num2str(size(index,1))]);
      disp(['negative svs:',num2str(size(sv,1))]);

      HW2_Utils.genRsltFile(w, b, 'val','./data/result_val');
      [ap, prec, rec]=HW2_Utils.cmpAP('./data/result_val','val');
      close();
      disp(['ap:',num2str(ap)]);
      
      ws=[ws,w];
      bs=[bs,b];
      objs=[objs,obj];
      aps=[aps,ap];
   
end
    
    fig1=figure(1);
    plot([1:k+1],objs,'-ro');
    legend('objective values');
    saveas(fig1,'objective.png');
    
    fig2=figure(2);
    plot([1:k+1],aps,'-ro');
    legend('ap');
    saveas(fig2,'ap.png');
    
%     HW2_Utils.genRsltFile(w, b, 'test','./data/result_test');
    

