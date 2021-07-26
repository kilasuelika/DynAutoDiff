function m=inv_vstack_half(x, d)
m=zeros(d,d);
k=0;
for i=1:d
	m(i:d,i)=x((k+1):(k+(d-i+1)));
	k=k+d-i+1;
end
end