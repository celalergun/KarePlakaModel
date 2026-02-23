#include "mainwindow.h"
#include <QFileDialog>
#include "./ui_mainwindow.h"
#include <iostream>

using std::cout;
using std::endl;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QString modelPath = QCoreApplication::applicationDirPath() + QDir::separator() + "model" + QDir::separator() + "kareplaka.onnx";
    plateFinder = new PlateFinder(modelPath);
}

MainWindow::~MainWindow()
{
    delete ui;
}

QImage MainWindow::putImage(const cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS=1
    if(mat.type()==CV_8UC1)
    {
        // Set the color table (used to translate colour indexes to qRgb values)
        QVector<QRgb> colorTable;
        for (int i=0; i<256; i++)
            colorTable.push_back(qRgb(i,i,i));
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        img.setColorTable(colorTable);
        return img;
    }
    // 8-bits unsigned, NO. OF CHANNELS=3
    if(mat.type()==CV_8UC3)
    {
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}

void MainWindow::on_btnResim_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("All files"), QDir::homePath(), tr("All files (*.*)"));
    if( !filename.isEmpty() )
    {
        cout << "Dosya: " << filename.toStdString() << endl;
        cv::Mat image = cv::imread(filename.toStdString(), cv::IMREAD_COLOR);
        ui->lblResim->setPixmap(QPixmap::fromImage(putImage(image)));
        auto plates = plateFinder->InspectPicture(image, 0.90f);
    }
}

