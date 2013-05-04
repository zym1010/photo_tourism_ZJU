
[pointMat, vectorList] = PositionList;


starts = pointMat;
ends = vectorList;


quiver3(starts(:,1), starts(:,2), starts(:,3), ends(:,1), ends(:,2), ends(:,3))
axis equal;

labels = num2str((1:9)','%d');   
text(pointMat(:,1), pointMat(:,2),pointMat(:,3), labels, 'horizontal','left', 'vertical','bottom');

