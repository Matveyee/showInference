#include "../include/camera.hpp"

AICamera::AICamera() {
    
    int status = GXInitLib();

    if (status != GX_STATUS_SUCCESS) {
        std::cout << "Library initialization failed : ";
        std::cout << status << std::endl;
        return;
    }

    //Updates the enumeration list for the devices.
    uint32_t nDeviceNum = 0;
    status = GXUpdateDeviceList(&nDeviceNum, 1000);
    if ((status != GX_STATUS_SUCCESS) || (nDeviceNum <= 0)) {
        std::cout << "Update list failed" << std::endl;
        return;
    }
    //Opens the device.
    status = GXOpenDeviceByIndex(1, &hDevice);

    if (status != GX_STATUS_SUCCESS) {
        std::cout << "Failed opening" << std::endl;
        return;
    }

    status = GXSetEnum(hDevice, GX_ENUM_LINE_SELECTOR,
    GX_ENUM_LINE_SELECTOR_LINE2);
    
    status = GXSetEnum(hDevice, GX_ENUM_LINE_MODE,
    GX_ENUM_LINE_MODE_INPUT);

    status = GXSetEnum(hDevice, GX_ENUM_TRIGGER_SOURCE,
    GX_TRIGGER_SOURCE_LINE2);

    //Sets the trigger mode to ON.
    status = GXSetEnum(hDevice, GX_ENUM_TRIGGER_MODE,
    GX_TRIGGER_MODE_ON);
    //Sets the trigger activation mode to the rising edge.
    status = GXSetEnum(hDevice,GX_ENUM_TRIGGER_ACTIVATION,
    GX_TRIGGER_ACTIVATION_RISINGEDGE);

        GX_FLOAT_RANGE raisingRange;
    status = GXGetFloatRange(hDevice,GX_FLOAT_TRIGGER_FILTER_RAISING,
    &raisingRange);
    //Sets the rising edge filter to the minimum value.
    status = GXSetFloat(hDevice, GX_FLOAT_TRIGGER_FILTER_RAISING,
    raisingRange.dMin);
    std::cout << "Min " << raisingRange.dMin << std::endl;

}
void AICamera::setCallBack(GXCaptureCallBack cllbck) {
    GXRegisterCaptureCallback(hDevice, NULL, cllbck);
}

void AICamera::setWidth(int w) {
    width = w;
    GXSetInt(hDevice, GX_INT_WIDTH, w);
}

void AICamera::setHeight(int h) {
    height = h;
    GXSetInt(hDevice, GX_INT_HEIGHT, h);
}

void AICamera::setOffsetX(int x) {
    offsetX = x;
    GXSetInt(hDevice, GX_INT_OFFSET_X, x);
}

void AICamera::setOffsetY(int y) {
    offsetY = y;
    GXSetInt(hDevice, GX_INT_OFFSET_Y, y);
}

void AICamera::startCapture() {
    GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_START);
}

void AICamera::stopCapture() {
    GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_STOP);

    GXUnregisterCaptureCallback(hDevice);
    GXCloseDevice(hDevice);
    GXCloseLib();
}