%% Function to compute discriminant function
function g = discriminant_func(alpha, p, C, sigma, train_data, train_label, data, b)

if (p == 1 && C == 10e6)
    w = sum(alpha .* train_label .* (train_data' * data));
    g = w + b;
elseif sigma ~= 0
    w = sum(alpha .* train_label .* (exp(-1 * (dist(train_data.', data)) / (2 * sigma^2))));
    g = w + b;
else
    w = sum(alpha .* train_label .* ((train_data' * data) + 1).^p);
    g = w + b;
end

end