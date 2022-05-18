function Mueller_Euler()
%% graphics
N = 100;
XMIN = -1.5; XMAX = 1.5;
YMIN = -0.5; YMAX = 2;
t1 = linspace(XMIN,XMAX,N);
t2 = linspace(YMIN,YMAX,N);
[x, y] = meshgrid(t1, t2);
V = mueller([x(:),y(:)]);
figure;
contour(x,y,reshape(V,N,N),-200:10:400,'linewidth',1,'color','k')
hold on
grid;
%% parameters for time stepping
dt = 1e-5; % time step
Temp = 30; % temperature
sqh = sqrt(dt*2*Temp);
N = 10^6; %length of full trajectory
subsample = 10^2;
traj = zeros(floor(N/subsample),2);
x0 = [0,0];
X = x0;
j = 0;
k = 0;
while j < N
    dVmueller = grad_mueller(X);
    X = X - dt*dVmueller + sqh*randn(1,2);
    if mod(j, subsample) == 0
        traj(k+1,:) = X;
        k = k + 1;
    end
    j = j + 1;
end
% save('Muller_trajectory_beta_inv_30.mat', 'traj');
% save('metad_bias_trajectory_seed1.mat', 'static_traj');
%%
figure;
hold on;
contour(x,y,reshape(V,100,100),-200:10:400,'linewidth',1,'color','k')
grid;
scatter(traj(:,1), traj(:,2),10, 'filled');
end
%%
%%
function V = mueller(x)
a = [-1,-1,-6.5,0.7];
b = [0,0,11,0.6];
c = [-10,-10,-6.5,0.7];
D = [-200,-100,-170,15];
X = [1,0,-0.5,-1];
Y = [0,0.5,1.5,1];
[t,~]=size(x);
V=zeros(t,1);
for i = 1 : 4
    Vnew = D(i)*exp(a(i)*(x(:,1)-X(i)).^2+b(i)*(x(:,1)-X(i)).*(x(:,2)-Y(i))+...
        c(i)*(x(:,2)-Y(i)).^2);
    V=V+Vnew;
end
        
end

%%
function dV = grad_mueller(x)
a = [-1,-1,-6.5,0.7];
b = [0,0,11,0.6];
c = [-10,-10,-6.5,0.7];
D = [-200,-100,-170,15];
X = [1,0,-0.5,-1];
Y = [0,0.5,1.5,1];
[t,~]=size(x);
dV=zeros(t,2);
for i = 1 : 4
    Vnew = D(i)*exp(a(i)*(x(:,1)-X(i)).^2+b(i)*(x(:,1)-X(i)).*(x(:,2)-Y(i))+...
        c(i)*(x(:,2)-Y(i)).^2);
    dVnew(:,1)=(2*a(i)*(x(:,1)-X(i))+b(i)*(x(:,2)-Y(i))).*Vnew;
    dVnew(:,2)=(2*c(i)*(x(:,2)-Y(i))+b(i)*(x(:,1)-X(i))).*Vnew;
    dV=dV+dVnew;
end
        
end



