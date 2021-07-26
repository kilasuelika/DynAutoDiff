function fv=nll(x,data)
d=size(data,2);
d2=d*(d+1)/2;

p=x(1);
p=exp(p)/(1+exp(p));

mu1=x(2:(d+1));
mu2=x((d+2):(2*d+1));

S1=inv_vstack_half(x((2*d+2):(2*d+1+d2)),d);
S1=S1'*S1+eye(10);
S2=inv_vstack_half(x((2*d+1+d2+1):end),d);
S2=S2'*S2+eye(10);

%eig(S1)
%eig(S2)
%p1v=mvnpdf(data,mu1,S1);
%for i=1:5
%	data(i,:)-mu1
%end
%mat2str(p1v(1:5))
fvs=-log(p*mvnpdf(data,mu1,S1)+(1-p)*mvnpdf(data,mu2,S2));
%num2str(fvs(1),15)
fv=sum(fvs);
end