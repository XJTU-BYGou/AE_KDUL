function [temp_vecT,temp_vecTR] = dataoperator(res,operatnumber)

if operatnumber == 0    
    for n = 1:length(res)
        temp_res = res(n);
        %% Convert the BLOB into a type single
        temp_vecSamples = single(typecast(temp_res.Data,'int16'));
        %% finally convert data into engineering units
        temp_vecTR{n} = temp_vecSamples' * temp_res.TR_mV;
        temp_vecT{n} = (1: length(temp_vecTR{n}))/temp_res.SampleRate;
    end
elseif operatnumber == 1  
        columnnumber = 0;
        fname = fieldnames(res);
    for i = length(fname):-1:1
        if ismember(i,columnnumber) == 1
            continue
        else
            temp_vecT = rmfield(res,fname{i});
        end
    end
end
        

end