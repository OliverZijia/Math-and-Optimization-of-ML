clc
clear
close all
%% create the data
N = 25;
X = reshape(linspace(0,0.9,N),[N,1]);
Y = cos(10*X.^2) + 0.1 * sin(100*X);

%% MLE of trigonometrc
% order = 1
fi1 = [ones(25,1),sin(2 * pi .* X), cos(2 * pi .* X)];
omega1 = inv(fi1' * fi1) * (fi1' * Y);

%order = 11
fi11 = [ones(25,1), sin(2 * pi .* X), cos(2 * pi .* X), sin(4 * pi .* X),cos(4 * pi .* X),sin(6 * pi .* X), cos(6 * pi .* X),sin(8 * pi .*X), cos(8 * pi .* X), sin(10 * pi .* X), cos(10 * pi .* X), sin(12 * pi .* X),cos(12 * pi .* X),sin(14 * pi .* X), cos(14 * pi .* X),sin(16 * pi .* X), cos(16 * pi .* X), sin(18 * pi .* X),cos(18 * pi .* X),sin(20 * pi .* X), cos(20 * pi .* X),sin(22 * pi .* X), cos(22 * pi .* X)];
omega11 = inv(fi11' * fi11) * (fi11' * Y);

%% plot part
x = reshape(linspace(-0.3,1.3,200),[200,1]);
Fi1 = [ones(200,1),sin(2 * pi .* x), cos(2 * pi .* x)];
Fi11 = [ones(200,1), sin(2 * pi .* x), cos(2 * pi .* x), sin(4 * pi .* x),cos(4 * pi .* x),sin(6 * pi .* x), cos(6 * pi .* x), sin(8 * pi .*x), cos(8 * pi .* x), sin(10 * pi .* x), cos(10 * pi .* x), sin(12 * pi .* x),cos(12 * pi .* x),sin(14 * pi .* x), cos(14 * pi .* x),sin(16 * pi .* x), cos(16 * pi .* x), sin(18 * pi .* x),cos(18 * pi .* x),sin(20 * pi .* x), cos(20 * pi .* x),sin(22 * pi .* x), cos(22 * pi .* x)];
y1 = Fi1 * omega1;
y11 = Fi11 * omega11;

figure
plot(X,Y,'ro', x,y1, x,y11)
legend('Original Data','Order = 1', 'Order = 11')
axis([-0.3 1.3 -1.5 2.0])
title('MLE of trignometric Regression with different order')
xlabel('x')
ylabel('predictive mean')


