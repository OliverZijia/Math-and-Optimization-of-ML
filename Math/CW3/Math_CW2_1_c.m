clear all
clc
close all
%% create the data
N = 25;
X = reshape(linspace(0,0.9,N),[N,1]);
Y = cos(10*X.^2) + 0.1 * sin(100*X);
summation = zeros(11,1);
MSE = zeros(11,1);
sigma_ML = zeros(11,1);

%% sigma of ML
Fi_ml = [ones(25,1), sin(2 * pi .* X), cos(2 * pi .* X), sin(4 * pi .* X),cos(4 * pi .* X),sin(6 * pi .* X), cos(6 * pi .* X),sin(8 * pi .*X), cos(8 * pi .* X), sin(10 * pi .* X), cos(10 * pi .* X), sin(12 * pi .* X),cos(12 * pi .* X),sin(14 * pi .* X), cos(14 * pi .* X),sin(16 * pi .* X), cos(16 * pi .* X), sin(18 * pi .* X),cos(18 * pi .* X),sin(20 * pi .* X), cos(20 * pi .* X),sin(22 * pi .* X), cos(22 * pi .* X)];
    for i = 1:11
        fi_ml = Fi_ml(:,1:(2*i-1));
        omega = inv(fi_ml' * fi_ml) * (fi_ml' * Y);
        sigma_ML(i,1) = sum((Y - fi_ml * omega).^2)/25;
    end

%% leave-one-out cross validation
indices=crossvalind('Kfold',25,25);
    for k=1:25
        test = (indices == k); 
        train = ~test;
        train_data= X(train,:);
        train_target= Y(train,:);
        test_data= X(test,:);
        test_target= Y(test,:);
        Fi_train = [ones(24,1), sin(2 * pi .* train_data), cos(2 * pi .* train_data), sin(4 * pi .* train_data),cos(4 * pi .* train_data),sin(6 * pi .* train_data), cos(6 * pi .* train_data),sin(8 * pi .*train_data), cos(8 * pi .* train_data), sin(10 * pi .* train_data), cos(10 * pi .* train_data), sin(12 * pi .* train_data),cos(12 * pi .* train_data),sin(14 * pi .* train_data), cos(14 * pi .* train_data),sin(16 * pi .* train_data), cos(16 * pi .* train_data), sin(18 * pi .* train_data),cos(18 * pi .* train_data),sin(20 * pi .* train_data), cos(20 * pi .* train_data)];       
        Fi_test = [ones(1,1), sin(2 * pi .* test_data), cos(2 * pi .* test_data), sin(4 * pi .* test_data),cos(4 * pi .* test_data),sin(6 * pi .* test_data), cos(6 * pi .* test_data),sin(8 * pi .*test_data), cos(8 * pi .* test_data), sin(10 * pi .* test_data), cos(10 * pi .* test_data), sin(12 * pi .* test_data),cos(12 * pi .* test_data),sin(14 * pi .* test_data), cos(14 * pi .* test_data),sin(16 * pi .* test_data), cos(16 * pi .* test_data), sin(18 * pi .* test_data),cos(18 * pi .* test_data),sin(20 * pi .* test_data), cos(20 * pi .* test_data)]; 
        for j = 1:11
            fi_train = Fi_train(:,1:(2*j-1));
            omega1 = inv(fi_train' * fi_train) * (fi_train' * train_target);
            fi_test = Fi_test(:,1:(2*j-1));
            y1 = fi_test * omega1;
            summation(j,1) = summation(j,1) + (test_target - y1)^2;
        end
    end
MSE = summation ./ 25;   

%% plot part
order = linspace(0,10,11);
figure
plot(order',MSE,order',sigma_ML)
legend('Leave-One-out','Maximum Likelihood')
%axis([0 10 0 5])
title('MSE ofLeave-one-out cross validation versus MLE')
xlabel('order')
ylabel('MSE')


