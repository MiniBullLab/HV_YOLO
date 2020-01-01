import sys, os, shutil, time
from distutils.core import setup
from Cython.Build import cythonize

start_time = time.time()
curr_dir = os.path.abspath('.')
parent_path = sys.argv[1] if len(sys.argv) > 1 else ""
setup_file = __file__.replace('/', '\\')
build_dir = "build"
build_tmp_dir = build_dir + "/temp"

s = "# cython: language_level=3"


def get_py(base_path=os.path.abspath('.'), parent_path='', excepts=()):
    """
    获取py文件的路径
    :param base_path: 根路径
    :param parent_path: 父路径
    :param excepts: 排除文件
    :return: py文件的迭代器
    """
    full_path = os.path.join(base_path, parent_path, name)
    for filename in os.listdir(full_path):
        full_filename = os.path.join(full_path, filename)
        if os.path.isdir(full_filename) and filename != build_dir and not filename.startswith('.'):
            for f in get_py(base_path, os.path.join(parent_path, name), filename, excepts, copyOther, delC):
                yield f
        elif os.path.isfile(full_filename):
            ext = os.path.splitext(filename)[1]
            if ext == ".c":
                if delC and os.stat(full_filename).st_mtime > start_time:
                    os.remove(full_filename)
            elif full_filename not in excepts and os.path.splitext(filename)[1] not in ('.pyc', '.pyx'):
                if os.path.splitext(filename)[1] in ('.py', '.pyx') and not filename.startswith('__'):
                    path = os.path.join(parent_path, name, filename)
                    yield path
        else:
            pass


def pack_pyd():
    # 获取py列表
    module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,)))
    try:
        setup(
            ext_modules=cythonize(module_list),
            script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir],
        )
    except Exception as ex:
        print("error! ", str(ex))
    else:
        module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,), copyOther=True))

    module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,), delC=True))
    if os.path.exists(build_tmp_dir):
        shutil.rmtree(build_tmp_dir)

    print("complate! time:", time.time() - start_time, 's')

def delete_c(path='.', excepts=(setup_file,)):
    '''
    删除编译过程中生成的.c文件
    :param path:
    :param excepts:
    :return:
    '''
    dirs = os.listdir(path)
    for dir in dirs:
        new_dir = os.path.join(path, dir)
        if os.path.isfile(new_dir):
            ext = os.path.splitext(new_dir)[1]
            if ext == '.c':
                os.remove(new_dir)
        elif os.path.isdir(new_dir):
            delete_c(new_dir)


if __name__ == '__main__':
    try:
        pack_pyd()
    except Exception as e:
        print(str(e))
    finally:
        delete_c()
