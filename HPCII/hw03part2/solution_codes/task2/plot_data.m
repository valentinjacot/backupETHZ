
A = importdata( 'data.in', ' ',1); 
d = A.data;

x=d(:,1); y=d(:,2);z=d(:,3);

tri = delaunay(x,y);

h = trisurf(tri, x, y, z);

axis vis3d

%% Clean it up

% axis off
l = light('Position',[-50 -15 29]);
% set(gca,'CameraPosition',[208 -50 7687])
lighting phong
shading interp
% colorbar EastOutside
