function [X, out] = lq_pgd_mc(q, Y, P, lamda, Xtrue, X0);

[m,n] = size(Y);

% Lipschitz constant
L = 1.001;

%Convergence setup
MAX_ITER = 2e3;
ABSTOL = 1e-7;

%Initialize
if nargin<6
	X = zeros(m,n);
else
    X = X0;
end

out.et = [];out.e = [];
num_stop = 0;

for iter = 1 : MAX_ITER

    Xm1 = X;	

    [S,V,D] = svd(X - (1/L)*((X - Y).*P));
    v = shrinkage_Lq(diag(V), q, lamda, L);
    indx = find(v>0);
    X = S(:,indx)*diag(v(indx))*D(:,indx)';
    
    out.e  = [out.e, norm(X-Xm1,'fro')/norm(Xtrue,'fro')];
        
    %Check for convergence
    if (norm(X-Xm1,'fro')< sqrt(m*n)*ABSTOL) 
        num_stop = num_stop + 1;
        if num_stop==3
            break;
        end
    else
        num_stop = 0;
    end

end


