function [w, b, objective,index] = SVM_QP(x,y,C)

    [n,d]=size(x);

    if(n ~= size(y,1))
        disp('size is not correspondent');
    end
        
    f=-ones(n,1);
    A=[];
    b=[];
    Aeq=y';
    beq=0;
    lb=zeros(n,1);
    ub=C*ones(n,1);
    epsilon=C*1e-6;
    H=(y*y').*(x*x');
%   H = H + 1e-10*eye(n);

    [alpha, obj] = quadprog(H,f,A,b,Aeq,beq,lb,ub);


    % objective value
    objective=-obj;
%     disp(['objective value: ',num2str(objective)]);
    
    % find coefficients
    w=((alpha.*y)'*x)';
    
    % find support vectors
    index=find(alpha>epsilon&alpha<C-epsilon);
    sv=[index,x(index,:)];
%     disp(['number of support vectors: ',num2str(size(index,1))]);
    
    % find offset
    b=y(index)-x(index,:)*w;
    b=min(b);
    
end