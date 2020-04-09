clear all
clc
X=1:5;
L=length(X);
Y=X'.*X';
Xii = repmat(X.*X,L,1);
Xjj=Xii';

XXi_XXj = Xii+Xjj-2*X'*X;
X=X';
sigma_f = 1;
sigma_l = 6;
sigma_n = -Inf;1;
eta=1e-2;
err=1; tol=1e-3;
count=0;



f = @(x)logPosterior(x,X,Y);
options = optimoptions('fminunc');
options.MaxFunctionEvaluations=1e4;
[sol,res] = fminunc(f,[5,61,0.01],options)

P = fitrgp(X,Y);

P


% while (err>tol)
% %     if count>400
% %         break
% %     end
%     Kp = exp(sigma_f)*exp(-0.5*exp(sigma_l)*XXi_XXj);
%     Q = Kp + exp(sigma_n)*eye(L,L);
%     Qinv=inv(Q);
%     det(Q)
%     dPdf = 0.5*Y'*Qinv*Kp*Qinv*Y - 0.5*trace(Qinv*Kp);
%     dPdl = 0.5*Y'*Qinv*(Kp.*(-0.5*exp(sigma_l)*XXi_XXj))*Qinv*Y - 0.5*trace(Qinv*(Kp.*(-0.5*exp(sigma_l)*XXi_XXj)));
%     dPdn = 0.5*Y'*Qinv*exp(sigma_n)*eye(L,L)*Qinv*Y - 0.5*trace(Qinv*exp(sigma_n)*eye(L,L));
%     err = abs(dPdl)+abs(dPdf);
%     
%     
%     sigma_l = sigma_l-eta*dPdl;
%     sigma_f = sigma_f-eta*dPdf;
% %     sigma_n = sigma_n-eta*dPdn;
%     count=count+1;
%     
% end
