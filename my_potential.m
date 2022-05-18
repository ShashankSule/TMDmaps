function dV = grad_potential(x)
mu = [-1., 0 ; 1, 0.];  % gaussian means
c_inv = [2., 0.; 0. 1.]; % gaussian inverse covariance
energy = 10.0;

[N, ~] = size(x);
dV = zeros(N, 2);
for i=1:2
    z = (x - mu(i, :));
    mat = z*c_inv;
    e = exp(-diag(z*c_inv*z'));
    dV = dV - 2*e*mat;
end
dV = -energy*dV;
dV(:, 1) = dV(:, 1) + 4*x(:, 1).^3;
dV(:, 2) = dV(:, 2) + 4*x(:, 2).^3;
end
