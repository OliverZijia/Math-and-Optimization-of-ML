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

%% plot part
x = reshape(linspace(-0.3,1.3,200),[200,1]);
Gauss = ones(200,21);
for k = 1:20
    Gauss(:,k+1) = gaussmf(x,[scale mean(1,k)]);
end
figure
plot(X,Y,'ro')
legend('origin data')
hold on
for n = 1:10
    y = Gauss * omega(:,n);
    plot(x,y)
%    legend('lemada = %f',lemada(1,n))
end
%legend('Original Data','Order = 1', 'Order = 11')
%axis([-0.3 1.3 -1.5 2.0])
title('Radge Regresion of 20 Gaussian with different lemada')
xlabel('x')
ylabel('predictive mean')
