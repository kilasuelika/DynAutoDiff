function x=vstack_half(m, d)
x=zeros(d*(d+1)/2,1);
k=0;
for i=1:d
	for j=1:i
		k=k+1;
		x(k)=m(i,j);
	end
end
end