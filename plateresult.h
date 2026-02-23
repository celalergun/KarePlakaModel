#ifndef PLATERESULT_H
#define PLATERESULT_H

struct PlateResult
{
    int x, y, w, h;
    float confidence;
    int objclass;
};

#endif // PLATERESULT_H
