function res_tra = importSpecialWaveformData(res_pri,filepath,databasename)

C0 = [' ('];

dataDepth = 999;
N = ceil(length(res_pri)/dataDepth);


if length(res_pri) <dataDepth+1

for i = 1:length(res_pri)
    ID = res_pri(i).TRAI;
    if i > 1
        C = [C,' or TRAI=',num2str(ID)];     
    else
        C = [C0,' TRAI=',num2str(ID)];
    end
end
C = [C,')'];
res_tra = importaedata(fullfile(filepath,databasename),0,['WHERE  ',C],0);

else

res_tra = [];
for i = 1:N
    if i < N
        for n = (i-1)*dataDepth+1 : i*dataDepth
            ID = res_pri(n).TRAI;
            if mod(n,dataDepth) ~= 1
                C = [C,' or TRAI=',num2str(ID)];     
            else
                C = [C0,' TRAI=',num2str(ID)];
            end            
        end
    else
        for n = (i-1)*dataDepth+1 : length(res_pri)
            ID = res_pri(n).TRAI;
            if mod(n,dataDepth) ~= 1
                C = [C,' or TRAI=',num2str(ID)];     
            else
                C = [C0,' TRAI=',num2str(ID)];
            end            
        end
    end
    C = [C,')'];
    tmp_res_tra = importaedata(fullfile(filepath,databasename),0,['WHERE  ',C],0);
    res_tra = [res_tra;tmp_res_tra];
end

end
end