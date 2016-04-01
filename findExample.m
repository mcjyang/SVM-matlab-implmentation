function [feature,label, rects] = findExample(num,w,b,iter,ubAnno)
    

    rects=[];
    for i=1:93
        ubs=ubAnno{i};
        im = imread(sprintf('%s/trainIms/%04d.jpg', HW2_Utils.dataDir, i));
        tem=HW2_Utils.detect(im,w,b,0);
        for j=1:size(tem,2)
            badIdx(1,j) = or(tem(3,j) > size(im,2), tem(4,j) > size(im,1));
        end
        tem = tem(:,~badIdx);
        badIdx=[];
        for k=1:size(ubs,2)
            overlap = HW2_Utils.rectOverlap(tem, ubs(:,k)); 
            tem = tem(:, overlap < 0.3);
        end
        
        rects=[rects,[tem;i*ones(1,size(tem,2))]];
%         disp(['detecting training picture:',num2str(i)]);
    end
    
    rects=sortrows(rects',-5)';
    rects=int64(rects);
    fea_HOG=[];

   
    for i=1:num
        im = imread(sprintf('./data/trainIms/%04d.jpg',rects(6,i)));
        imReg = im(rects(2,i):rects(4,i), rects(1,i):rects(3,i),:);
        imReg = imresize(imReg, HW2_Utils.normImSz);
        fea_HOG{i}=HW2_Utils.cmpFeat(rgb2gray(imReg)); 
    end    
 
    
    n=size(fea_HOG,2);
    fea_HOG = cat(2, fea_HOG{:});
    feature=HW2_Utils.l2Norm(fea_HOG);
    label=-ones(n,1);

end