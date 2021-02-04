function d = distance_plane(x_start, y_start, x_tar, y_tar)

%Euclidean distance (L2 norm)
d = sqrt((x_start - x_tar)^2 + (y_start - y_tar)^2);

%Manhattan distance
%d = (x_start - x_tar) + (y_start - y_tar);

end

