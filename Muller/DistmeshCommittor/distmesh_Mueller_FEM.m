function distmesh_Mueller_FEM()
close all
fsz = 16;
%% the displayed domain:
XMIN = -1.5; XMAX = 1;
YMIN = -0.5; YMAX = 2;
%% the computational domain
bbxmin = -4; bbxmax = 2;
bbymin = -2; bbymax = 4;
bbox = [bbxmin,bbymin;bbxmax,bbymax];
res = 100;
xv = linspace(bbxmin,bbxmax,res); yv = linspace(bbymin,bbymax,res);
[xx,yy] = meshgrid(xv,yv);
v = reshape(mueller([xx(:),yy(:)]),res,res);
figure;
hold on
h = contour(xv,yv,v,[-200 : 25 : 1600]);
h1 = contour(xv,yv,v,[1600,1600],'r','LineWidth',4);
h2 = contour(xv,yv,v,[-100,-100],'r','LineWidth',4);
grid
set(gca,'Fontsize',20);
colorbar
xlabel('x','Fontsize',20);
ylabel('y','Fontsize',20);
%%
figure;
hold on;
% get boundary points
a = [-0.558;1.441]; % center of A
b = [0.623;0.028]; % center of B
r = 0.1;
Vbdry = 100;
t = linspace(0,2*pi,20)'; % adjust this one to vary mesh size 
t(end) = [];
pA = [a(1)+r*cos(t),a(2)+r*sin(t)];
pB = [b(1)+r*cos(t),b(2)+r*sin(t)];
VA = max(mueller(pA));
VB = max(mueller(pB));
Vthreshold = 0.5*(Vbdry + max(VA,VB));
hV = contour(xx,yy,v,[Vbdry,Vbdry]);
hV(:,1) = [];
hV = hV';
h = r*(t(2)-t(1));
hV = reparametrization(hV,h);
bdry = [pA;pB;hV];
plot(bdry(:,1),bdry(:,2),'.','Markersize',15);
drawnow;



fd = @(p)ddiff(mueller(p)-Vbdry,dunion(dcircle(p,a(1),a(2),r),dcircle(p,b(1),b(2),r)));
[pts,tr] = distmesh2d(fd,@huniform,h,bbox,bdry);

%% choose number of mesh refinements
nrefine = 1;
for i = 1 : nrefine 
    [pts,tr] = refine(pts,tr);
end
%% smooth mesh
[pts,tr] = smoothmesh(pts,tr);
triplot(tr,pts(:,1),pts(:,2),'color','r');
%% Start processing mesh
%% extract boundary nodes and boundary edges
% Find boundary nodes
edges = [tr(:,1),tr(:,2);tr(:,2),tr(:,3);tr(:,1),tr(:,3)];
e0 = edges;
edges = sort(edges,2);
[~,isort2] = sort(edges(:,2),'ascend');
edges = edges(isort2,:);
[~,isort1] = sort(edges(:,1),'ascend');
edges = edges(isort1,:);
Nedges = length(edges);
fprintf('N edges = %i, N points = %i\n',length(edges),length(pts));
% boundary edges are encountered only once as they belong to only one
% triangle
[uedges,ishrink,~] = unique(edges,'rows');
Nis = length(ishrink);
fprintf('N ishrink = %i\n',Nis);
gap = circshift(ishrink,[-1,0]) - ishrink;
if ishrink(end) == Nedges
    gap(end) = 1;
else
    gap(end) = [];
end    
i1 = find(gap == 1);
n1 = length(i1);
i2 = find(gap == 2);
n2 = length(i2);
fprintf('n1 = %i, n2 = %i\n',n1,n2);
ie1 = ishrink(i1);
bedges = edges(ie1,:);
Nbe = size(bedges,1);
ind = (1 : Nedges)';
ie2 = ind;
ie2(ie1) = [];
Nie = length(ie2);
fprintf('Nbe = %i, Nie = %i\n',Nbe,Nie);
figure; hold on
for i = 1 : Nbe
    j = bedges(i,:);
    plot([pts(j(1),1),pts(j(2),1)],[pts(j(1),2),pts(j(2),2)],'Linewidth',2);
end
bb = [bedges(:,1);bedges(:,2)];
ibnodes = unique(bb); % indices of boundary nodes
Nib = length(ibnodes);
bnodes = pts(ibnodes,:); % boundary nodes
plot(bnodes(:,1),bnodes(:,2),'.','Markersize',10);
drawnow;
Nbdry = size(bnodes,1);
% To use Tarjan's algorithm, we need the vertices to be indexed from 1 to
% Nverts
Npts = size(pts,1);
map = zeros(Npts,1);
map(ibnodes) = (1 : Nib)';
E = [map(bedges(:,1)),map(bedges(:,2));map(bedges(:,2)),map(bedges(:,1))];
% Tarjan's algorithm
NSCC = Tarjan([1 : Nib]',E);
col = parula(NSCC + 1);
for i = 1 : NSCC
    fname = sprintf('SCC%d',i);
    SCC = load(fname);
    ind = SCC.SCC;
    plot(pts(ibnodes(ind),1),pts(ibnodes(ind),2),'.','Markersize',10,'color',col(i,:));
    drawnow;
    fprintf('Boundary: SCC %d: %i nodes\n',i,length(ind));
end
daspect([1,1,1])
set(gca,'Fontsize',20);
triplot(tr,pts(:,1),pts(:,2));
% %% find neighbors
% Ned = length(uedges);
% nneib = zeros(Npts,1);
% nei = zeros(Npts);
% for j = 1 : Ned
%     v = uedges(j,1);
%     w = uedges(j,2);
%     nneib(v) = nneib(v) + 1;
%     nneib(w) = nneib(w) + 1;
%     nei(v,nneib(v)) = w;
%     nei(w,nneib(w)) = v;
% end
% maxnneib = max(nneib);
%% find boundary nodes for dirichlet0, dirichlet1, and neumann boundaries
for i = 1 : NSCC
    fname = sprintf('SCC%d',i);
    SCC = load(fname);
    ind = SCC.SCC;

    iinit = ibnodes(ind);
    vrts_init = pts(iinit,:);
    
    % check if this connected component corresponds to the outer boundary    
    if min(mueller(vrts_init)) > Vthreshold
        neumann = iinit;
    else % this is a Dirichlet boundary
        if max(abs(vrts_init(:,1)-a(1))<r & abs(vrts_init(:,2)-a(2))<r)
            dirichlet0 = iinit;
        else
            dirichlet1 = iinit;
        end
    end
end
figure;
hold on
triplot(tr,pts(:,1),pts(:,2),'color','k');
Nbe = length(bedges);
for j = 1 : Nbe
    plot([pts(bedges(j,1),1),pts(bedges(j,2),1)],[pts(bedges(j,1),2),pts(bedges(j,2),2)],'Linewidth',2);
end
daspect([1,1,1])
%% end processing mesh
% call FEM
[u,u1,A1,b1] = FEM2d(pts,tr,neumann',[dirichlet0',dirichlet1'],dirichlet1');

committor = full(u);
% graphic representation
figure(3)
clf;
trisurf(tr,pts(:,1),pts(:,2),full(u),'facecolor','interp')
view(2)
colorbar

figure(4)
clf;
trisurf(tr,pts(:,1),pts(:,2),full(u),'facecolor','interp')
view(2)
colorbar
axis([XMIN,XMAX,YMIN,YMAX]);
set(gca,'Fontsize',16);
daspect([1,1,1]);
save('DistmeshMueller.mat','pts','tr','committor');
end

%% FEM2d
function [u,u1,A1,b1] = FEM2d(coordinates,elements3,neumann,dirichlet,dirichlet1)
FreeNodes=setdiff(1:size(coordinates,1),dirichlet);
A = sparse(size(coordinates,1),size(coordinates,1));
b = sparse(size(coordinates,1),1);

% Assembly
for j = 1:size(elements3,1)
  A(elements3(j,:),elements3(j,:)) = A(elements3(j,:),elements3(j,:)) ...
      + stima3(coordinates(elements3(j,:),:)) * ...
      exp(-0.05*mypot(sum(coordinates(elements3(j,:),:),1)/3));
end

% Volume Forces
for j = 1:size(elements3,1)
  b(elements3(j,:)) = 0;  
end

% Neumann conditions
for j = 1 : size(neumann,1)
  b(neumann(j,:))=b(neumann(j,:)) + norm(coordinates(neumann(j,1),:)- ...
      coordinates(neumann(j,2),:)) * myg(sum(coordinates(neumann(j,:),:))/2)/2;
end

% Dirichlet conditions 
u = sparse(size(coordinates,1),1);
u(dirichlet1) = 1;
b = b - A * u;

% Computation of the solution
u(FreeNodes) = A(FreeNodes,FreeNodes) \ b(FreeNodes);

A1 = A(FreeNodes,FreeNodes);
b1 = b(FreeNodes);
u1 = u(FreeNodes);
end


%%
function M = stima3(vertices)
d = size(vertices,2);
G = [ones(1,d+1);vertices'] \ [zeros(1,d);eye(d)];
M = det([ones(1,d+1);vertices']) * G * G' / prod(1:d);
end

%%
function Vrm = mypot(x)
gamma=9;
k=5;
V = mueller(x);
Vrm = V; % + gamma*sin(2*k*pi*x(:,1)).*sin(2*k*pi*x(:,2));
end

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
function Stress = myg(x)
Stress = zeros(size(x,1),1);
end
%%
function path = reparametrization(path,h)
    %reparametrization
    dp = path - circshift(path,[1,0]);
    dp(1,:) = 0;
    dl = sqrt(sum(dp.^2,2));
    lp = cumsum(dl);
    len = lp(end);
    lp = lp/len; % normalize
    npath = round(len/h);
    g1 = linspace(0,1,npath)';
    path = interp1(lp,path,g1);
end