function [w,b] = SVM_SGD(x,y,C)
    
    [n,d]=size(x);
    if(n ~= size(y,1))
        disp('size is not correspondent');
    end
    
    epoch=1000;
    
    eta_1=100;
    eta_0=1;
    w=zeros(d,1);
    b=sum(y-x*w)/n;
    losses=[];
    
    for i=1:epoch
        
        eta=eta_0/(eta_1+epoch);
        tem=[x,y];
        tem=tem(randperm(size(tem,1)),:);
        x=tem(:,1:d);
        y=tem(:,d+1);
        
        loss=[];
        
        for k=1:n
            w_old=w;
            b_old=b;
            yk=y(k,1);
            xk=x(k,:);
            c=yk*(xk*w+b);
            if(c<1)
                w=w+eta*yk*xk';
                b=b+eta*yk;
                loss=[loss;1-c];
            else
                w=w;
                b=b;
                loss=[loss;0];
            end 
        end
        
        obj=w'*w/(2*n)+C*sum(loss);
        losses=[losses;obj];
        
    end
    
    disp(['objective value is ',num2str(obj)]);
    
    fig=figure(1);
    plot([1:epoch],losses,'-ro');
    legend('Loss after each epoch');
    saveas(fig,'Loss_SVM_SGD_100.png');

end