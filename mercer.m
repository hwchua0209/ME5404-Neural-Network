%% Function to compute eigenvalue of data for Mercer's condition
function [K, H] = mercer(p, C, sigma, data, label)

n = size(data, 2);

K = zeros(n);
H = zeros(n);
for i = 1:n
    for j = 1:n
        if (p == 1 && C == 10e6)
            K(i, j) = dot(data(:, i), data(:, j));
        elseif sigma ~= 0
            K(i, j) = exp(-1 * (norm(data(:, i) - data(:, j))) / (2 * sigma^2));
        else
            K(i, j) = (dot(data(:, i), data(:, j)) + 1)^p;
        end
        H(i, j) = label(i) * label(j) * K(i, j);
    end
end
e = eig(K);

if min(e) < -1e-4
    warning('Kernel is not admissible!');
else
    disp('Kernel is admissible.');
end   
end