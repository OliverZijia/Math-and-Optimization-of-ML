clc
clear
close all
%% create the data
N = 25;
X = reshape(linspace(0,0.9,N),[N,1]);
Y = cos(10*X.^2) + 0.1 * sin(100*X);

%% MLE of Polynomial
% order = 0
fi0 = ones(25,1);
omega0 = (fi0' * Y)/(fi0' * fi0);

% order = 1
fi1 = [ones(25,1),X];
omega1 = inv(fi1' * fi1) * (fi1' * Y);

%order = 2
fi2 = [ones(25,1), X, X.^2];
omega2 = inv(fi2' * fi2) * (fi2' * Y);

%order = 3
fi3 = [ones(25,1), X, X.^2, X.^3];
omega3 = inv(fi3' * fi3) * (fi3' * Y);

%order = 11
fi11 = [ones(25,1), X, X.^2, X.^3, X.^4, X.^5, X.^6, X.^7, X.^8, X.^9, X.^10];
omega11 = inv(fi11' * fi11) * (fi11' * Y);

%% plot part
x = reshape(linspace(-0.3,1.3,200),[200,1]);
Fi0 = ones(200,1);
Fi1 = [ones(200,1),x];
Fi2 = [ones(200,1),x, x.^2];
Fi3 = [ones(200,1),x, x.^2, x.^3];
Fi11 = [ones(200,1), x, x.^2, x.^3, x.^4, x.^5, x.^6, x.^7, x.^8, x.^9, x.^10];
y0 = Fi0 * omega0;
y1 = Fi1 * omega1;
y2 = Fi2 * omega2;
y3 = Fi3 * omega3;
y11 = Fi11 * omega11;

figure
plot(X,Y,'ro', x,y0, x,y1, x,y2, x,y3, x,y11)
legend('Original Data','Order = 0','Order = 1', 'Order = 2', 'Order = 3', 'Order = 11')
axis([-0.3 1.3 -1.5 2.0])
title('MLE of Polynomial Regression with different order')
xlabel('x')
ylabel('predictive mean')


