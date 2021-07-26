function report_sol(x,d)
d2=d*(d+1)/2;

p=x(1);
p=exp(p)/(1+exp(p));

mu1=x(2:(d+1));
mu2=x((d+2):(2*d+1));

S1=inv_vstack_half(x((2*d+2):(2*d+1+d2)),d);
S1=S1'*S1;
S2=inv_vstack_half(x((2*d+1+d2+1):end),d);
S2=S2'*S2;

fprintf("p1: %f, p2:%f\n",p,1-p);
fprintf("mu1:\n");
disp(mu1);
fprintf("mu2:\n");
disp(mu2);
fprintf("Sigma1:\n");
disp(S1);
fprintf("Sigma1:\n");
disp(S2);
end