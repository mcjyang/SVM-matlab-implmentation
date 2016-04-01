clc; close all; clear all;

data= load('q3_1_data.mat');
tX=data.trD';
tY=data.trLb;
vX=data.valD';
vY=data.valLb;

C=0.1;
% C=100;

% Quadprog Programming
[w,b]=SVM_QP(tX,tY,C);
y=vX*w+b;

y(find(y>0))=1;
y(find(y<=0))=-1;
result=[vY,y];
correct=size(find(vY.*y>0),1);
accuracy=correct/size(vY,1)
disp(['accuracy: ',num2str(accuracy)]);
confusion=[size(find(vY>0&y>0),1),size(find(vY<0&y>0),1);
           size(find(vY>0&y<0),1),size(find(vY<0&y<0),1)]


       
% SGD
[w,b]=SVM_SGD(tX,tY,C);

y=vX*w+b;
y(find(y>0))=1;
y(find(y<=0))=-1;
result=[vY,y];
error=size(find(vY.*y<0),1);
error_test=error/size(vY,1);
disp(['test error: ',num2str(error),',',num2str(size(vY,1))]);

y=tX*w+b;
y(find(y>0))=1;
y(find(y<=0))=-1;
result=[tY,y];
error=size(find(tY.*y<0),1);
error_train=error/size(tY,1);
disp(['train error: ',num2str(error),',',num2str(size(tY,1))]);

norm_w=sum(abs(w));
disp(['norm w: ',num2str(norm_w)]);


