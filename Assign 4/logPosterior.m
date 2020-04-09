function logP = logPosterior(theta,X,Y)

sigma_f=theta(1);sigma_l=theta(2);sigma_n=theta(3);

m=length(X);

Xii = repmat(X.*X,1,m);
Xjj=Xii';

XXi_XXj = Xii+Xjj-2*X*X';


% Kp = exp(sigma_f)*exp(-0.5*exp(sigma_l)*XXi_XXj);
Kp = sigma_f^2*exp(-0.5*sigma_l^2*XXi_XXj) ;
K=Kp+ sigma_n^2*eye(m,m)

det(K)
% Kp=XXi_XXj;
% L = chol(Kp);
% c1 = L\Y;%solve(L,Y);
% beta = L'\c1;%solve(L',c1);

% logP = -0.5*Y'*beta-sum(log(diag(L))) - size(X,1)/2*log(2*pi);


logP = -0.5*Y'*(K\Y)-0.5*sum(log(diag(K))) - size(X,1)/2*log(2*pi);


end