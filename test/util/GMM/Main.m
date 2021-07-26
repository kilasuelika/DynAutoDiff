rng(4);

data=zeros(5000,10);
mu1=randn(1,10);
Sigma1=randn(10,10);
Sigma1=Sigma1'*Sigma1+eye(10);
mu2=randn(1,10)+5;
Sigma2=randn(10,10)*1.5;
Sigma2=Sigma2'*Sigma2+eye(10);

for i=1:size(data,1)
	if rand()<0.25
		data(i,:)=mvnrnd(mu1, Sigma1);
	else
		data(i,:)=mvnrnd(mu2,Sigma2);
	end
end
writematrix(data, "../../gmm_datat5000_10.txt",'Delimiter',' ');

S1Init=randn(10,10);
S2Init=randn(10,10);
writematrix(S1Init,"../../S1Init.txt",'Delimiter',' ');
writematrix(S2Init,"../../S2Init.txt",'Delimiter',' ');

%%
data=readmatrix("../../gmm_datat5000_10.txt",'Delimiter',' ');
m1=readmatrix("../../S1Init.txt");
m2=readmatrix("../../S2Init.txt");

d=size(data,2);
d2=d*(d+1)/2;

x=zeros(1,1+2*d+d*(d+1));
x(1)=0.5;
x(2:(1+d))=10;
x(d+2:(2*d+1))=-10;

m1=m1*m1';
x((2*d+2):(2*d+1+d2))=vstack_half(m1,10);
m2=m2*m2';
x((2*d+1+d2+1):end) = vstack_half(m2,10);

A = [];
b = [];
Aeq = [];
beq = [];
lb=[];
ub=[];
nonlcon=[];

options = optimoptions('fmincon','Display','iter',...
	'MaxFunctionEvaluations',Inf,...
	'MaxIterations',Inf,'UseParallel',true);
objfun=@(x) nll(x,data);
sol=fmincon(objfun,x, A,b, Aeq, beq,lb,ub,nonlcon,options);
report_sol(sol,d);

%%
clc
options = optimoptions('fminunc','Display','iter',...
	'MaxFunctionEvaluations',Inf,...
	'MaxIterations',Inf,'UseParallel',true);
objfun=@(x) nll(x,data);
sol=fminunc(objfun,x,options);
report_sol(sol,d);

%%
%grad
x_g=zeros(1, length(x));
for i=1:length(x)
	xp=x;
	xb=x;
	xp(i)=x(i)+1e-4;
	xb(i)=x(i)-1e-4;
	x_g(i)=(nll_test(xp, data)-nll_test(xb,data))/(1e-4*2);
end
mat2str(x_g)
 mat2str(x_g((2*d+2):(2*d+1+d2)))
 
%%
x_g=zeros(1, length(x));
for i=1:length(x)
	xp=x;
	xb=x;
	xp(i)=x(i)+1e-4;
	xb(i)=x(i)-1e-4;
	x_g(i)=(nll(xp,data)-nll(xb,data))/(1e-4*2);
end
mat2str(x_g)
 mat2str(x_g((2*d+2):(2*d+1+d2)))
