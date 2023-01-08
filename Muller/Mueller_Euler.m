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
dt = 1e-4; % time step
Temp = 1; % temperature
sqh = sqrt(dt*2*Temp);
N = 10^6; %length of full trajectory
subsample = 10^3;
traj = zeros(floor(N/subsample),2);
x0 = [0.25,0.50];
X = x0;
j = 0;
k = 0;
figure; 
hold on;
contour(x,y,reshape(V,100,100),-200:50:400,'linewidth',1,'color','k')
set(gca, 'YTickLabel', [])
set(gca, 'XTickLabel', [])
set(gca, 'YTick', [])
set(gca, 'XTick', [])
xlim([-1.5, 1.5])
ylim([-0.5,2])
grid;
while j < N
    dVmueller = grad_mueller(X);
    X = X - dt*dVmueller + sqh*randn(1,2);
%     s = scatter(X(1), X(2),50, 'filled');
%     s.MarkerFaceAlpha = 0.6;
%     drawnow;
%     hold on;
    if mod(j, subsample) == 0
        traj(k+1,:) = X;
        % s = scatter(X(1), X(2),40, 'filled');
        % s.MarkerFaceAlpha = 0.6;
        % drawnow;
        % hold on;
        k = k + 1;
    end
    j = j + 1;
end
% save('Muller_trajectory_beta_inv_30.mat', 'traj');
% save('metad_bias_trajectory_seed1.mat', 'static_traj');
%%
figure;
hold on;
contour(x,y,reshape(V,100,100),-200:20:400,'linewidth',1,'color','k')
% grid;
p = plot(traj(:,1), traj(:,2), '-', 'LineWidth', 5);
p.Color(4) = 0.2;      
color = turbo(k); 
s = scatter(traj(:,1), traj(:,2), 100, color, 'filled', 'MarkerFaceAlpha',0.6);
xlim([-0.5 1]);
ylim([-0.5 1]);

%% gif plotter
figure;
hold on;
contour(x,y,reshape(V,100,100),-200:10:400,'linewidth',1,'color','k')
xlim([-0.5 1])
ylim([-0.5 1])
for i=1:1000
    p = plot(traj(1:i,1),traj(1:i,2), 'ro-'); 
    drawnow;
end
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

