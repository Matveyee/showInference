#include "GxIAPI.h"
#include <iostream>


#ifndef CAMERA.HPP
#define CAMERA.HPP
// Abstract class, with callback function type as a parameter
template<typename CallBackType>
class AbstractCamera {

    public:
        
        int width;
        int height;
        int offsetX = 0;
        int offsetY = 0;

        virtual void setCallBack(CallBackType cllbck) = 0;

        virtual void setWidth(int w) = 0;
        
        virtual void setHeight(int h) = 0;
        
        virtual void setOffsetX(int x) = 0;
        
        virtual void setOffsetY(int y) = 0;

        virtual void startCapture() = 0;

        virtual void stopCapture() = 0;


};

class AICamera : AbstractCamera<GXCaptureCallBack> {

    public:

        GX_DEV_HANDLE hDevice;

        AICamera();

        void setCallBack(GXCaptureCallBack cllbck) override;

        void setWidth(int w) override;

        void setHeight(int h) override;

        void setOffsetX(int x) override;
        
        void setOffsetY(int y) override;

        void startCapture() override;

        void stopCapture() override;

};

#endif