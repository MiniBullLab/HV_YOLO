# 代码打包
将python代码的一个包，打包为一个可安装的库

#### easyai包打包
1. ``` cp -r ../easyai . ```
2. ``` python3 py2sec.py -d easyai -m __init__.py,setup.py,easy_ai.py```（该步骤执行时间会比较长几分钟）
3. ``` cd result```
4. 将最外层目录中的LICENSE、MANIFEST.in、README.md、setup.cfg、setup.py以及与该代码运动环境对应的requirements_xxx文件拷贝到result目录中，并将requirements_xxx文件的文件名修改为requirements。
5. ``` python3 setup.py bdist_wheel```
6. 在文件夹dist中将whl文件拷贝走，即为最后打包好的文件
7. 将py2so文件夹中生成的文件与拷贝过来的文件删除


