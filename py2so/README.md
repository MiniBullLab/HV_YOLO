# 代码打包
将python代码的一个包，打包为一个可安装的库

#### easyai包打包
1. ``` cp -r ../easyai . ```
2. ``` python3 py2sec.py -d easyai -m __init__.py,setup.py,easy_ai.py```（该步骤执行时间会比较长几分钟）
3. ``` cd result```
4. 将ai_build目录中的全部文件拷贝到result目录中，如果需要打包对应版本的库，可以将最外层中的requirements_xxx文件的文件名修改为requirements，替换当前requirements文件。
5. ``` python3 setup.py bdist_wheel```
6. 在文件夹dist中将whl文件拷贝走，即为最后打包好的文件
7. 将py2so文件夹中生成的文件与拷贝过来的文件删除

#### easy_convert包打包
1. ``` cp -r ../easy_convert . ```
2. ``` python3 py2sec.py -d easy_convert -m __init__.py,setup.py,easy_convert.py ```（该步骤执行时间会比较长几分钟）
3. ``` cd result```
4. 将convert_build目录中的全部文件拷贝到result目录中，如果需要打包对应版本的库，可以将最外层中的requirements_xxx文件的文件名修改为requirements，替换当前requirements文件。
5. ``` python3 setup.py bdist_wheel```
6. 在文件夹dist中将whl文件拷贝走，即为最后打包好的文件
7. 将py2so文件夹中生成的文件与拷贝过来的文件删除
