function TwoWell_Euler()
%% graphics
N = 100;
XMIN = -1.5; XMAX = 1.5;
YMIN = -1.5; YMAX = 1.5;
t1 = linspace(XMIN,XMAX,N);
t2 = linspace(YMIN,YMAX,N);
[x, y] = meshgrid(t1, t2);
for i=1:N
    for j = 1:N
        V(i,j) = my_potential([x(i,j) y(i, j)]);
    end
end
figure;
contour(x,y,reshape(V,N,N),-15:0.5:10,'linewidth',1,'color','k')

hold on
grid;
%% parameters for time stepping
dt = 1e-2; % time step
Temp = 1.5; % temperature
sqh = sqrt(dt*2*Temp);
N = 1.5*10^4; %length of full trajectory
subsample = 1;
traj = zeros(floor(N/subsample),2);
x0 = [0 0];
X = x0;
j = 0;
k = 0;
while j < N
    dV = grad_potential(X);
    X = X - dt*dV + sqh*randn(1,2);
    if mod(j, subsample) == 0
        traj(k+1,:) = X;
        k = k + 1;
    end
    j = j + 1;
end
save('Twowell_trajectory_1.5.mat', 'traj');
% save('metad_bias_trajectory_seed1.mat', 'static_traj');
%%
figure;
hold on;
contour(x,y,reshape(V,100,100),-10:0.5:10,'linewidth',1,'color','k')
grid;
scatter(traj(:,1), traj(:,2),10, 'filled');
end
%%
%%
function V = my_potential(x)
mu = [-1, 0 ; 1, 0.]; % gaussian means
c_inv = [2., 0.; 0. 1.];    % gaussian inverse covariance
energy = 10.0;
my_sum = 0;
for i=1:2
    z = (x - mu(i, :));
    my_sum = my_sum + exp(-diag(z*(c_inv*z')));
end
V = -energy*my_sum + x(:, 1).^4 + x(:, 2).^4;
end

%%
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



