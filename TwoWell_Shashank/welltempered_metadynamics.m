% function welltempered_metadynamics()
%% Initialize Euler-Maruyama parameters
seed = 1;
rng(seed);
N = 1e6;  % number of data points
dt = 1e-2; % time step
x0 = [-1 0]';
X = x0;
subsample = 1e2;
Nsub = floor(N/subsample);
traj = zeros(2, Nsub);

% Set parameters for diffusion / nois
beta_inv = 2.0; % temperature
sqh = sqrt(dt*2*beta_inv);

% Initialize Metadynamics Parameters
height = 1.0;              % height of gaussian bias
stride_steps = 1000;         % Number of simulations steps per deposition
width = 0.01;                 % Widths / Covar matrix for collective vars
gamma = 10.0;            % fluctuation parameter for WTmetad
welltemp = 1/(beta_inv*(gamma - 1));
total_deps = floor(N/stride_steps);

m = 0;           % Initialize number of gaussians
scales = zeros(total_deps); % Initialize list of gaussian centers/means
centers = zeros(2, total_deps); % Initialize list of exp. scaling factors

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
contour(x,y,reshape(V,N,N),-15:0.5:1,'linewidth',1,'color','k')
hold on
grid;
%% Simulate Euler-Maruyama with Metadynamics biased potential
N = 1e6;
i = 1;
step = 0;
xval = traj(:, i);
while step < N
    %%% Update Trajectory %%
    % Update bias if we hit another stride step
    if mod(step, stride_steps) == 0
        m = m + 1;
        
        new_center = xval;
        diffs = reshape(new_center,2,1) - centers(:, 1:m);      % Note:only as many diffs as there are gaussians
        sqdists = sum(diffs.^2, 1);
        gaussian_vec = height*exp(-sqdists/(2.*width));
        wt_scales_vec = exp(-welltemp*scales(1:m));
        scales(m) = sum(gaussian_vec.*wt_scales_vec, 'all');
        centers(:, m) = new_center;
        fprintf("adding bias..%d\n" ,m)
        
        % Update bias gradient
        if m > 1
            diffs = reshape(xval,2,1) - centers(:, 1:m);      % Note:only as many diffs as there are gaussians
            sqdists = sum(diffs.^2, 1);
            gaussian_vec = height*exp(-sqdists/(2.*width));
            wt_scales_vec = exp(-welltemp.*scales(1:m));
            bias_grad = sum(-gaussian_vec.*(wt_scales_vec/width).*diffs, 2);
        else
            bias_grad = zeros(2 , 1);
        end
        
    end
    % Update Drift
    dV = grad_potential(xval'); 
    xval = xval - (dV' - bias_grad)*dt + sqh*randn(2,1);
    if mod(step, subsample) == 0
        X(:, i + 1) = xval;
        i = i + 1;
%         if mod(i, 1000) == 0
%             fprintf(i)
%             i = i + 1;
%             % step = step + 1;
%         end
    end
    step = step + 1;
    % plot(X(1,:), X(2,:), 'o')
    % hold on; 
    % drawnow;
end
% end
%% sample long trajectory 

N = 1e8;
traj = zeros(2,N+1); 
traj(:,1) = xval; 
subsample = 1e4; 
samples = zeros(2,floor(N/subsample)); 
j=1; 
for i=1:N
    dV = grad_potential(traj(:,i)'); 
    traj(:,i+1) = traj(:,i) - (dV' - bias_grad)*dt + sqh*randn(2,1);
    if mod(i, subsample) == 0
        samples(:, j) = traj(:,i);
        j = j + 1;
    end
end
    
    
    
    

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

