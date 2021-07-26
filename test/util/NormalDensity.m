x=[1,2,3];
y=[3,4,5];
Sigma=[[1,2,5];[2,5,7];[3,4,9]];

z1=x*inv(Sigma)*y'
Sigma1=Sigma;
Sigma1(1,1)=Sigma(1,1)+0.001;
z2=x*inv(Sigma1)*y'
(z2-z1)/0.001

analytic=x'*y.*(-inv(Sigma)*sone(3,1,1)*inv(Sigma));
sum(analytic,'all')

dS=inv(Sigma1)-inv(Sigma);
dS(1,1)/0.001
-inv(Sigma)*sone(3,1,1)*inv(Sigma)



%%
%x^TAx
z1=x*Sigma*y';
z2=x*Sigma1*y';
(z2-z1)/0.001

%%
X=[1,9;5,8;4,7];
mu=[2,5];
S=[2,1;1,2];

log(mvnpdf(X,mu,S))
ol=sum(log(mvnpdf(X,mu,S)));

%X
dX=zeros(3,2);
for i=1:3
    for j=1:2
        X1=X;
        X1(i,j)=X(i,j)+0.0001;
        dX(i,j)=(sum(log(mvnpdf(X1,mu,S)))-ol)/0.0001;
    end
end
dX
%mu
dmu=zeros(1,2);
for i=1:2
    mu1=mu;
    mu1(i)=mu(i)+0.0001;
    dmu(i)=(sum(log(mvnpdf(X,mu1,S)))-ol)/0.0001;
end
dmu
%Sigma
dS=zeros(2);
for i=1:2
    for j=1:2
        S1=S;
        S1(i,j)=S(i,j)+0.00001;
        S1(j,i)=S1(i,j);
        dS(i,j)=(sum(log(mvnpdf(X,mu,S1)))-ol)/0.00001;
    end
end
dS