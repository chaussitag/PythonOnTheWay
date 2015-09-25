#!/usr/bin/env python
#coding=utf8

import argparse
import os
import shutil
import sys
import textwrap
import time

class BasicProcessor(object):
    def __init__(self, srcDir, target):
        self.__srcDir = None
        self.__target = None

        self.srcDir = srcDir
        self.target = target

    @property
    def srcDir(self):
        return self.__srcDir
    @srcDir.setter
    def srcDir(self, srcDir):
        self.__srcDir = srcDir

    @property
    def target(self):
        return self.__target
    @target.setter
    def target(self, target):
        self.__target = target

    @staticmethod
    def createFile(path, content):
        with open(path, "w") as fileObject:
            fileObject.write(content)

    def preProcess(self):
        pass

    def postProcess(self):
        self.generateProjectProperties()
        self.generateDotClasspath()

    def process(self):
        self.preProcess()
        shutil.copytree(self.srcDir, self.target)
        self.postProcess()

    def generateProjectProperties(self):
        content = self.getProjectPropertiesContent()
        if (len(content) > 0):
            self.createFile(self.target + "/project.properties", content)

    def getProjectPropertiesContent(self):
        pass

    def generateDotClasspath(self):
        content = self.getDotClasspathContent()
        if (len(content) > 0):
            self.createFile(self.target + "/.classpath", content)

    def getDotClasspathContent(self):
        pass

    def generateManifest(self):
        self.createFile(self.target + "/AndroidManifest.xml", self.getManifestContent())

    def getManifestContent(self):
        content='''\
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
          package="org.appspot.apprtc"
          android:versionCode="1"
          android:versionName="1.0">
          <uses-sdk android:minSdkVersion="14" android:targetSdkVersion="21" />
</manifest>
'''
        return content

class AppRTCDemoProcessor(BasicProcessor):
    def __init__(self, srcDir, target, webrtcsrc):
        super(AppRTCDemoProcessor, self).__init__(srcDir, target)
        self.__webrtcsrc = webrtcsrc

    ## readonly property
    @property
    def webrtcsrc(self):
        return self.__webrtcsrc

    def postProcess(self):
        super(AppRTCDemoProcessor, self).postProcess()

        ## copy the libjingle_peerconnection_so.so
        self.copyShareLibrary()

        ## change the default room server configuration
        cmd = r"""\
        sed -i 's|<string name="pref_room_server_url_default".*</string>|<string name="pref_room_server_url_default" translatable="false">http://223.167.80.29:8081</string>|' \
        """ + self.target + "/res/values/strings.xml"
        os.system(cmd)


    def copyShareLibrary(self):
        if (os.path.isdir(self.target + "/libs") != True):
            os.mkdir(self.target + "/libs")

        out = "out"
        if (os.path.isdir(self.webrtcsrc + "/" + out) != True):
            out = "out_android"
        if (os.path.isdir(self.webrtcsrc + "/" + out) != True):
            print("could not locate the out directory, failed to copy libjingle_peerconnection_so.so, you should copy it your self")
            sys.exit(1)

        buildMode = "Debug"
        if (os.path.isdir(self.webrtcsrc + "/" + out + "/" + buildMode) != True):
            buildMode = "Release"
        if (os.path.isdir(self.webrtcsrc + "/" + out + "/" + buildMode) != True):
            print("could not locate libjingle_peerconnection_so.so, you should copy it yourself")
            sys.exit(1)

        if (os.path.isfile(self.webrtcsrc + "/" + out + "/" + buildMode + "/AppRTCDemo/libs/armeabi-v7a/libjingle_peerconnection_so.so") \
                != True):
            print("could not locate libjingle_peerconnection_so.so, you should copy it yourself")
            sys.exit(1)

        shutil.copytree(self.webrtcsrc + "/" + out + "/" + buildMode + "/AppRTCDemo/libs/armeabi-v7a", self.target + "/libs/armeabi-v7a")

    def getProjectPropertiesContent(self):
        content = '''\
# This file is automatically generated by Android Tools.
# Do not modify this file -- YOUR CHANGES WILL BE ERASED!
#
# This file must be checked in Version Control Systems.
#
# To customize properties used by the Ant build system edit
# "ant.properties", and override values to adapt the script to your
# project structure.
#
# To enable ProGuard to shrink and obfuscate your code, uncomment this (available properties: sdk.dir, user.home):
#proguard.config=${sdk.dir}/tools/proguard/proguard-android.txt:proguard-project.txt

# Project target.
target=android-21

java.compilerargs=-Xlint:all -Werror
android.library.reference.1=../libjingle_peer_connection_java
android.library.reference.3=../video_capture_android
android.library.reference.4=../video_render_android
android.library.reference.2=../audio_device_android
'''
        return content

    def getDotClasspathContent(self):
        content = '''\
<?xml version="1.0" encoding="UTF-8"?>
<classpath>
    <classpathentry kind="src" path="src"/>
    <classpathentry kind="src" path="gen"/>
    <classpathentry kind="con" path="com.android.ide.eclipse.adt.ANDROID_FRAMEWORK"/>
    <classpathentry exported="true" kind="con" path="com.android.ide.eclipse.adt.LIBRARIES"/>
    <classpathentry exported="true" kind="con" path="com.android.ide.eclipse.adt.DEPENDENCIES"/>
    <classpathentry exported="true" kind="lib" path="third_party/autobanh/autobanh.jar"/>
    <classpathentry kind="output" path="bin/classes"/>
</classpath>
'''
        return content

class SimpleLibraryProcessor(BasicProcessor):
    def __init__(self, srcDir, target):
        super(SimpleLibraryProcessor, self).__init__(srcDir, target)

    def postProcess(self):
        super(SimpleLibraryProcessor, self).postProcess()
        self.generateManifest()

    def getProjectPropertiesContent(self):
        content = '''
# This file is automatically generated by Android Tools.
# Do not modify this file -- YOUR CHANGES WILL BE ERASED!
#
# This file must be checked in Version Control Systems.
#
# To customize properties used by the Ant build system edit
# "ant.properties", and override values to adapt the script to your
# project structure.
#
# To enable ProGuard to shrink and obfuscate your code, uncomment this (available properties: sdk.dir, user.home):
#proguard.config=${sdk.dir}/tools/proguard/proguard-android.txt:proguard-project.txt

# Project target.
target=android-21
android.library=true
'''
        return content

    def getDotClasspathContent(self):
        content = '''\
<?xml version="1.0" encoding="UTF-8"?>
<classpath>
    <classpathentry kind="src" path="src"/>
    <classpathentry kind="src" path="gen"/>
    <classpathentry kind="con" path="com.android.ide.eclipse.adt.ANDROID_FRAMEWORK"/>
    <classpathentry exported="true" kind="con" path="com.android.ide.eclipse.adt.LIBRARIES"/>
    <classpathentry exported="true" kind="con" path="com.android.ide.eclipse.adt.DEPENDENCIES"/>
    <classpathentry kind="output" path="bin/classes"/>
</classpath>
'''
        return content


class LibjingleProcessor(SimpleLibraryProcessor):
    def __init__(self, srcDir, target):
        super(LibjingleProcessor, self).__init__(srcDir, target)

    def getDotClasspathContent(self):
        content = '''\
<?xml version="1.0" encoding="UTF-8"?>
<classpath>
    <classpathentry kind="src" path="src"/>
    <classpathentry kind="src" path="android"/>
    <classpathentry kind="src" path="gen"/>
    <classpathentry kind="con" path="com.android.ide.eclipse.adt.ANDROID_FRAMEWORK"/>
    <classpathentry exported="true" kind="con" path="com.android.ide.eclipse.adt.LIBRARIES"/>
    <classpathentry exported="true" kind="con" path="com.android.ide.eclipse.adt.DEPENDENCIES"/>
    <classpathentry combineaccessrules="false" kind="src" path="/audio_device_android"/>
    <classpathentry combineaccessrules="false" kind="src" path="/video_capture_android"/>
    <classpathentry combineaccessrules="false" kind="src" path="/video_render_android"/>
    <classpathentry kind="output" path="bin/classes"/>
</classpath>
'''
        return content


def getUsage():
    hint = """
    生成webrtc自带的android demo(AppRTCDemo)的eclipse工程, 生成后的目录可以直接导入eclipse里面进行编译调试。
    如果参数--webrtcsrc省略，则要把该脚本放到webrtc/src目录下;
    如果参数--outdir省略，就会生成到HOME目录
    """
    return textwrap.dedent(hint)


if __name__ == "__main__":
    outdir = "%s/webrtc_android_demo" % (os.getenv("HOME"), )
    webrtcsrc="."
    parser = argparse.ArgumentParser(description=getUsage())
    parser.add_argument("--outdir", "-o", default=os.getenv("HOME")+"/webrtc_android_demo",
                        help="存放生存的eclipse工程的目录位置")
    parser.add_argument("--webrtcsrc", "-r",
                        help="webrtc源码src目录的路径")
    args = parser.parse_args()

    webrtcsrc = args.webrtcsrc
    if (webrtcsrc == "" or os.path.isfile(webrtcsrc + "/build/android/envsetup.sh") != True):
        print("i don't know where is your webrtc/src directory, you can specify it by --webrtcsrc")
        parser.print_help()
        sys.exit(1)

    outdir = args.outdir
    if (os.path.isdir(outdir) == True):
        renameTarget = outdir + "-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
        print("%s exist, rename it to %s" % (outdir, renameTarget))
        shutil.move(outdir, renameTarget)

    os.makedirs(outdir)

    projects = [
        SimpleLibraryProcessor(webrtcsrc + "/webrtc/modules/audio_device/android/java",
                               outdir + "/audio_device_android"),
        SimpleLibraryProcessor(webrtcsrc + "/webrtc/modules/video_capture/android/java",
                               outdir + "/video_capture_android"),
        SimpleLibraryProcessor(webrtcsrc + "/webrtc/modules/video_render/android/java",
                               outdir + "/video_render_android"),
        LibjingleProcessor(webrtcsrc + "/talk/app/webrtc/java",
                           outdir + "/libjingle_peer_connection_java"),
        AppRTCDemoProcessor(webrtcsrc + "/talk/examples/android",
                            outdir + "/AppRTCDemo", webrtcsrc),
    ]

    for prj in projects:
        prj.process()
    print("the extracted project now located in %s" % (outdir, ))
