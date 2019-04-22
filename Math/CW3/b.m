clc
clear
% TODO: display the starting point and maximum 
%% create the date
N = 25;
X = reshape(linspace(0, 0.9, N), [N, 1]);
Y = cos(10*X.^2) + 0.1 * sin(100*X);

one = ones(25,1);
Phi = [one, X];

%% initialize the parameters
tol = 0.0001;
num_iter = 100000;
step_size = 0.001;

guess = [0.8,0.5];
alpha = guess(1,1);
beta = guess(1,2);

[n,m] = size(Phi);
I1 = eye(m);
I2 = eye(n);
k = alpha * Phi * I1 * Phi' + beta * I2;
l = Phi * Phi';

guess_eval  = -n/2 * log(2 * pi) - 1/2 * log(det(k)) - 1/2 * Y' * inv(k) * Y;
grad = [-0.5 * trace(inv(k) * l) + 0.5 * Y' * inv(k) * l * inv(k) * Y, -0.5 * trace(inv(k)) + 0.5 * Y' * inv(k) * inv(k) * Y];


% Store beta guesses at each iteration
guess_iter(1,:) = guess; 

% Store the function value at each iteration
fcn_val_iter(1)      = guess_eval;  

fprintf('\niter=%d; Func Val=%f; FONC Residual=%f',...
        0, guess_eval, norm(grad));

%% gradient descent

% Iterative algorithm begins ---------------------------------------------
for i = 1:num_iter                        
    % Step for gradient descent ------------------------------------------
    % *** Insert gradient descent code here ***
    % ***                                   ***
    guess = guess_iter(i,:) + step_size * grad; 

    % Update with the new iteration -------------------------------------
    alpha = guess(1,1);
    beta = guess(1,2);
    k = alpha * Phi * I1 * Phi' + beta * I2;
    l = Phi * Phi';
    
    guess_eval  = -n/2 * log(2 * pi) - 1/2 * log(det(k)) - 1/2 * Y' * inv(k) * Y;
    grad = [-0.5 * trace(inv(k) * l) + 0.5 * Y' * inv(k) * l * inv(k) * Y, -0.5 * trace(inv(k)) + 0.5 * Y' * inv(k) * inv(k) * Y];
    guess_iter(i+1,:) = guess;
    fcn_val_iter(i+1) = guess_eval;
    % Check if it's time to terminate ------------------------------------

    % Check the FONC?
    % Store the norm of the gradient at each iteration
    convgsd(i) = norm(grad); % <-- Correct this!!
    
    % Check that the vector is changing from iteration to iteration?
    % Stores length of the difference between the current beta and the 
    % previous one at each iteration
    lenXsd(i)  = norm(guess_iter(i+1,:)-guess_iter(i,:)); % <-- Correct this!!
    
    % Check that the objective is changing from iteration to iteration?
    % Stores the absolute value of the difference between the current 
    % function value and the previous one at each iteration
    diffFsd(i) = abs(fcn_val_iter(i+1)-fcn_val_iter(i)); % <-- Correct this!!
    
    fprintf('\niter=%d; Func Val=%f; FONC Residual=%f; Sqr Diff=%f',...
            i, guess_eval, convgsd(i), lenXsd(i));
    
    % Check the convergence criteria?
    if (convgsd(i) <= tol)
        fprintf('\nFirst-Order Optimality Condition met\n');
        break; 
%     elseif (lenXsd(i) <= tol)
%         fprintf('\nExit: Design not changing\n');
%         break;
%     elseif (diffFsd(i) <= tol)
%         fprintf('\nExit: Objective not changing\n');
%         break;
    elseif (i + 1 >= num_iter)
        fprintf('\nExit: Done iterating\n');
        break;
    end
    
end


%% plot the contour
N1 = 25;
X1 = reshape(linspace(0, 0.9, N1), [N1, 1]);
Y1 = cos(10*X1.^2) + 0.1 * sin(100*X1);


one1 = ones(25,1);
Phi1 = [one1, X1];

NN = 200;
[n1,m1] = size(Phi1);
I11 = eye(m1);
I21 = eye(n1);
a1 = linspace(0.3,0.8,NN);
b1 = linspace(0.42,0.55,NN);
% x1 = alpha  x2 = beta
for i = 1 : NN
    x1 = a1(1,i);
    for j = 1 : NN
        x2 = b1(1,j);  
        k = x1 * Phi1 * I11 * Phi1' + x2 * I21;
        f(j,i)  = -n1/2 * log(2 * pi) - 1/2 * log(det(k)) - 1/2 * Y1' * inv(k) * Y1;
    end
end
figure(1); clf; 
%h = fcontour(f,[0.1 1 0.35 0.6]);
[A,B] = meshgrid(a1,b1);
[C,h] = contour(a1,b1,f,100,'ShowText','on','TextStepMode','manual','TextStep',10,'LabelSpacing',200)

xlabel('alpha') 
ylabel('beta')
clabel(C,h,'FontSize',7)
title('Contour of funtion(respect to alpha beta)')
axis equal; hold on
%pbaspect([1 2])

% plot current point
plot(guess_iter(:,1),guess_iter(:,2),'ko-')