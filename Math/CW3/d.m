clear all
clc
close all
%% create the data
N = 25;
X = reshape(linspace(0,0.9,N),[N,1]);
Y = cos(10*X.^2) + 0.1 * sin(100*X);

%% Radge Regression of Gaussian
%syntax: y = gaussmf(x,[scale mean])
%sigma(scale)=0.1, 20 means belong to [0,1]. lemada = sigma^2/b^2
I = eye(21);
scale = 0.1;
lemada = linspace(1,100,10);% 0.1 overfitting 1 fine  10 underfitting
gauss = ones(25,21);
mean = linspace(0,1,20);
omega = zeros(21,10);
for i = 1:20
    gauss(:,i+1) = gaussmf(X,[scale mean(1,i)]);
end
for j = 1:10
    omega(:,j) = inv(gauss' * gauss + 2 * lemada(1,j) * I) * (gauss' * Y);
end