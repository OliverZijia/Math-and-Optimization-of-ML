clc
clear
% syms x1 x2
% f = 1 - exp(-((x1-1)^2 + x2^2)) - exp(-(3*x1^2 - 2*x1*x2 - 2*x1 + 6*x2 +3*x2^2 +3)) + 0.1 * log((x1^2 + 0.01)*(x2^2 + 0.01)-(x1*x2)^2);
% g = gradient(f,[x1,x2]);

x0 = [1 -1]';
% maximum number of allowed iterations
maxiter = 50;
% step size ( 0.33 causes instability, 0.2 quite accurate)
alpha = 1;
% initialize gradient norm, optimization vector, iteration counter, perturbation
gnorm = inf; x = x0; niter = 0; dx = inf;
% define the objective function:
f = @(x1,x2) 1 - (exp(-(x1.^2 + x2.^2 - 2.*x1 + 1)) + exp(-(3.*x1.^2 + 3.*x2.^2 - 2*x1.*x2 - 2*x1 + 6*x2 + 3)) - 0.1*log((x1.^2+0.01).*(x2.^2+0.01) - (x1.*x2).^2));
%f = @(x1,x2) 1 - exp(-(x1.^2 + x2.^2 - 2.*x1 + 1)) + exp(-(3.*x1.^2 + 3.*x2.^2 - 2*x1.*x2 - 2*x1 + 6*x2 + 3)) - 0.1*log((x1.^2+0.01).*(x2.^2+0.01) - (x1.*x2).^2);% plot objective function contours for visualization:
figure(1); clf; 
h = fcontour(f,[-2 2 -2 1.5]);
h.LevelStep = 0.05;
xlabel('x(1)') 
ylabel('x(2)')
title('Contour of f3 (alpha = 0.1)')
axis equal; hold on

% redefine objective function syntax for use with optimization:
f2 = @(x) f(x(1),x(2));
niter = 1;
% gradient descent algorithm:
while niter <= maxiter
    % calculate gradient:
    g = grad(x);
    gnorm = norm(g);
    % take step:
    xnew = x - alpha*g;
    % check step
    if ~isfinite(xnew)
        display(['Number of iterations: ' num2str(niter)])
        error('x is inf or NaN')
    end
    % plot current point
    plot([x(1) xnew(1)],[x(2) xnew(2)],'ko-')
    refresh
    % update termination metrics
    niter = niter + 1;
    dx = norm(xnew-x);
    x = xnew; 
end
% define the gradient of the objective
function g = grad(x)
g = [exp(- x(1)^2 + 2*x(1) - x(2)^2 - 1)*(2*x(1) - 2) - exp(- 3*x(1)^2 + 2*x(1)*x(2) + 2*x(1) - 3*x(2)^2 - 6*x(2) - 3)*(2*x(2) - 6*x(1) + 2) - (2*x(1)*(x(2)^2 + 1/100) - 2*x(1)*x(2)^2)/(10*(x(1)^2*x(2)^2 - (x(1)^2 + 1/100)*(x(2)^2 + 1/100)))
       exp(- 3*x(1)^2 + 2*x(1)*x(2) + 2*x(1) - 3*x(2)^2 - 6*x(2) - 3)*(6*x(2) - 2*x(1) + 6) - (2*x(2)*(x(1)^2 + 1/100) - 2*x(1)^2*x(2))/(10*(x(1)^2*x(2)^2 - (x(1)^2 + 1/100)*(x(2)^2 + 1/100))) + 2*x(2)*exp(- x(1)^2 + 2*x(1) - x(2)^2 - 1)];
end