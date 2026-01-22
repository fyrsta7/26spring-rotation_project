temp_viz::CloudWidget::CloudWidget(InputArray _cloud, InputArray _colors)
{
    Mat cloud = _cloud.getMat();
    Mat colors = _colors.getMat();
    CV_Assert(cloud.type() == CV_32FC3 || cloud.type() == CV_64FC3 || cloud.type() == CV_32FC4 || cloud.type() == CV_64FC4);
    CV_Assert(colors.type() == CV_8UC3 && cloud.size() == colors.size());

    if (cloud.isContinuous() && colors.isContinuous())
    {
        cloud.reshape(cloud.channels(), 1);
        colors.reshape(colors.channels(), 1);
    }

    vtkIdType nr_points;
    vtkSmartPointer<vtkPolyData> polydata = CreateCloudWidget::create(cloud, nr_points);

    // Filter colors
    Vec3b* colors_data = new Vec3b[nr_points];
    NanFilter::copy(colors, colors_data, cloud);

    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
    scalars->SetNumberOfComponents (3);
    scalars->SetNumberOfTuples (nr_points);
    scalars->SetArray (colors_data->val, 3 * nr_points, 0);

    // Assign the colors
    polydata->GetPointData ()->SetScalars (scalars);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput (polydata);

    cv::Vec3d minmax(scalars->GetRange());
    mapper->SetScalarRange(minmax.val);
    mapper->SetScalarModeToUsePointData ();

    bool interpolation = (polydata && polydata->GetNumberOfCells () != polydata->GetNumberOfVerts ());

    mapper->SetInterpolateScalarsBeforeMapping (interpolation);
    mapper->ScalarVisibilityOn ();
        
    mapper->ImmediateModeRenderingOff ();
    
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, polydata->GetNumberOfPoints () / 10)));
    actor->GetProperty ()->SetInterpolationToFlat ();
    actor->GetProperty ()->BackfaceCullingOn ();
    actor->SetMapper (mapper);
    
    WidgetAccessor::setProp(*this, actor);
}
