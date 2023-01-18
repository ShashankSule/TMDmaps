% function samples = MuellerMetadynamics()
close all
%% parameters for metadynamics
Ndeposit = 1000;
Nbumps = 1000;
height = 1;
sig = 0.1;
sig2 = sig^2;
%% graphics
N = 100;
XMIN = -1.5; XMAX = 1.5;
YMIN = -1.5; YMAX = 1.5;
t1 = linspace(XMIN,XMAX,N);
t2 = linspace(YMIN,YMAX,N);
[x, y] = meshgrid(t1, t2);
V = my_potential([x(:), y(:)]); 
% for i=1:N
%     for j = 1:N
%         V(i,j) = my_potential([x(i,j) y(i, j)]);
%     end
% end
figure;
contour(x,y,reshape(V,N,N),-15:0.5:10,'linewidth',1,'color','k')
hold on
grid;
%% parameters for time stepping
h = 1e-2; % time step
Temp = 1.5; % temperature
sqh = sqrt(h*2*Temp);
% traj = zeros(Ndeposit+1,2);
subsample = 1e2;
samples = zeros(floor(Ndeposit*Nbumps/subsample),2); 
x0 = [0,0];
coef = zeros(Nbumps,1);
xbump = zeros(Nbumps,2);
i=1; 
for k = 1 : Nbumps
    traj = zeros(Ndeposit+1,2);`
    traj(1,:) = x0;    
    w = sqh*randn(Ndeposit,2);
    for j = 1 : Ndeposit
        dVmueller = grad_potential(traj(j,:));
        aux1 = traj(j,1)-xbump(:,1);
        aux2 = traj(j,2)-xbump(:,2);      
        dVbump = sum((coef*[1,1]).*[aux1,aux2].*(exp(-0.5*(aux1.^2+aux2.^2)/sig2)*[1,1]),1)/sig2;
        traj(j+1,:) = traj(j,:) - h*(dVmueller-dVbump) + w(j,:);
        % subsampling routine 
        if mod((k-1)*Ndeposit + j, subsample) == 0 
            samples(i,:) = traj(j,:);
            i = i+1; 
        end
    end
    % plot(traj(:,1),traj(:,2));
    % drawnow;
    % fprintf('k = %d\n',k);
    % deposit bump
    x0 = traj(j+1,:);
    xbump(k,:) = x0;
    coef(k) = height;
end
% plot(samples(:,1),samples(:,2), 'bo')
% save('Twowell_data_metadynamics_beta_0.66.mat', 'samples'); 
%% Sample the modified potential 
figure;
contour(x,y,reshape(V,N,N),-200:10:400,'linewidth',1,'color','k')
hold on
grid;
Ndeposit = 1e6; 
samples = zeros(1e4, 2); 
traj = zeros(Ndeposit+1,2);
traj(1,:) = x0;    
w = sqh*randn(Ndeposit,2);
i=1; 
for j = 1 : Ndeposit
    dVmueller = grad_potential(traj(j,:));
    aux1 = traj(j,1)-xbump(:,1);
    aux2 = traj(j,2)-xbump(:,2);      
    dVbump = sum((coef*[1,1]).*[aux1,aux2].*(exp(-0.5*(aux1.^2+aux2.^2)/sig2)*[1,1]),1)/sig2;
    traj(j+1,:) = traj(j,:) - h*(dVmueller-dVbump) + w(j,:);
    % subsampling routine 
    if mod(j, 100) == 0 
        samples(i,:) = traj(j,:);
        i = i+1; 
    end
%     plot(traj(1:j,1),traj(1:j,2), 'bo')
%     drawnow;
end
plot(samples(:,1),samples(:,2), 'bo')
drawnow;
save('Twowell_data_metadynamics_longsample_beta_0.66.mat', 'samples');
% end

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