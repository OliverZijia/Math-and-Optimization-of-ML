clc
clear
close all
%% create the data
N = 25;
X = reshape(linspace(0,0.9,N),[N,1]);
Y = cos(10*X.^2) + 0.1 * sin(100*X);

%% MLE of trigonometrc
%order = 10
Phi11 = [ones(25,1), sin(2 * pi .* X), cos(2 * pi .* X), sin(4 * pi .* X),cos(4 * pi .* X),sin(6 * pi .* X), cos(6 * pi .* X),sin(8 * pi .*X), cos(8 * pi .* X), sin(10 * pi .* X), cos(10 * pi .* X), sin(12 * pi .* X),cos(12 * pi .* X),sin(14 * pi .* X), cos(14 * pi .* X),sin(16 * pi .* X), cos(16 * pi .* X), sin(18 * pi .* X),cos(18 * pi .* X),sin(20 * pi .* X), cos(20 * pi .* X),sin(22 * pi .* X), cos(22 * pi .* X)];

for j = 1 : 11
    Phi = Phi11(:,1:(2*j-1));
    tol = 0.01;
    num_iter = 100000000;
    step_size = 0.000001;
    
    guess = [0.3,0.2];
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

    fprintf('\norder=%d;iter=%d; Func Val=%f; FONC Residual=%f',...
            i,0, guess_eval, norm(grad));
        
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

            fprintf('\norder=%d;iter=%d; Func Val=%f; FONC Residual=%f; Sqr Diff=%f',...
                    j,i, guess_eval, convgsd(i), lenXsd(i));

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
itertime(1,j) = i;
large(1,j) = guess_eval;      
end

xn = linspace(1,11,11);
plot(xn,large);

