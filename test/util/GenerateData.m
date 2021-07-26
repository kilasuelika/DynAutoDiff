%%
%Linear regression
rng(5)
X=randn(1000,5);
y=X*[1,2,3,4,5]'+0.5;
writematrix(X,"../X.txt", "Delimiter"," ");
writematrix(y,"../y.txt", "Delimiter"," ");

%%
%Logistic regression.
