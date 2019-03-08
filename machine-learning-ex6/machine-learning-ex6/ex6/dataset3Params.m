function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Creating multiplicative steps for C and sigma
C_step = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_step = [0.01 0.03 0.1 0.3 1 3 10 30];

preds_error = zeros( length(C_step), length(sigma_step) );

output_c_sigma = zeros( length(C_step) + length(sigma_step), 3 );

row = 1;


for c_i = 1:length(C_step)
    for s_i = 1:length(sigma_step)
        cTest = C_step(c_i);
        sigmaTest = sigma_step(s_i);
        
        % Model training
        model = svmTrain(X, y, cTest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
        
        % predictions from SVM trained model
        preds = svmPredict(model, Xval);
        
        % returns the prdiction error
        preds_error(c_i,s_i) = mean( double( preds ~= yval));
        
        % Error for each C and sigma combination
        output_c_sigma(1,:) = [preds_error(c_i,s_i), cTest, sigmaTest];
        
        row = row + 1; 
    end
end

lowest_error = sortrows(output_c_sigma, 1);

C = lowest_error(1,2);
sigma = lowest_error(1,3);


        
        
        
        
        




% =========================================================================

end
