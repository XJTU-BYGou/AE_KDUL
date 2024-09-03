function res = importaedata(filename,typeflag,opcond,outputInfo)
%
%2018-9-28 by Boyuan Gou
%MATLAB R2018a
% filename contains path of data basement.
% typeflag:
%       0:tradb
%       1:pridb
% opcond(string):
%       eg:'WHERE TRAI<50000'
%
if ((nargin > 4) || (nargin < 2))
   error('Too many or too few input arguments.');
elseif(nargin == 3)
    outputInfo = 1;
elseif(nargin == 2)
    opcond = '';
    outputInfo = 1;
end

if typeflag == 0
    extname = '.tradb';
    sqlquery_pre = 'Select * FROM view_tr_data ';
elseif typeflag == 1
    extname = '.pridb';
    sqlquery_pre = 'Select * FROM view_ae_data ';
else
    error('typeflag Error!');
end

%% establishing a connection to the database
dbid = mksqlite(0, 'open', [filename,extname]);

%% sending a query to the database
%sqlquery = 'Select Data, TR_mV, SampleRate FROM view_tr_data ';%WHERE TRAI=1
sqlquery = [sqlquery_pre,opcond];
res = mksqlite(dbid,sqlquery);

mksqlite(dbid, 'close')

if outputInfo == 1
fprintf('%i results total.\n',length(res));
end
end