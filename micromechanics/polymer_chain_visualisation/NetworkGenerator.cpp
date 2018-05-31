
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <tuple>
#include <vector>

#include <eigen3/Eigen/Dense>

#include <range/v3/core.hpp>

#include <range/v3/algorithm.hpp>

#include <vtkCell.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

// To compile use the following command
// clang++ -O3 -std=c++1z -I/usr/include/vtk NetworkGenerator.cpp -L/usr/lib64/vtk
// -lvtkCommonDataModel -lvtkCommonCore -lvtkIOXML

using Coordinate = Eigen::Vector3d;

class BoundingBox
{
public:
    BoundingBox(double x, double y, double z) : x(x), y(y), z(z) {}

    Coordinate const dimensions() const { return {x, y, z}; }

    bool isInBox(Coordinate const& c) const
    {
        return c(0) < x and c(1) < y and c(2) < z and c(0) > 0.0 and c(1) > 0.0 and c(2) > 0.0;
    }

protected:
    double x, y, z;
};

Eigen::Matrix3d rotationMatrix(double const angle, Eigen::Vector3d const& axis)
{
    // Apply the Rodrigues theorem to rotate the vector
    Eigen::Matrix3d K;
    K << 0.0, -axis(2), axis(1), axis(2), 0.0, -axis(0), -axis(1), axis(0), 0.0;

    // Rotation matrix
    return Eigen::Matrix3d::Identity() + std::sin(M_PI - angle) * K +
           (1.0 - std::cos(M_PI - angle)) * K * K;
}

int main()
{
    // Create a random network with cross linking
    BoundingBox box(1.0, 1.0, 1.0);

    auto const segments = 10'000; // Number of segments
    auto const length = 0.1;      // Length of the segment
    auto const chains = 1;        // Number of chains in network
    auto const bond_angle = 109.5 * M_PI / 180.0;

    std::vector<Coordinate> coordinates(segments + 1, Coordinate::Zero());

    // Take a guess at the first location of the starting point by instantiating
    // a random number generator
    std::random_device rd;  // Used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> random_angle_gen(0.0, 2.0 * M_PI);

    auto const dim = box.dimensions();

    // Find a viable starting point for the chain
    coordinates[0] = Coordinate::Random().normalized() * length;

    // Randomly choose a point in a sphere of radius segment length
    // and create the first polymer segment
    auto const inclination = random_angle_gen(gen);
    auto const azimuth = random_angle_gen(gen);

    coordinates[1] = Coordinate(length * std::sin(inclination) * std::cos(azimuth),
                                length * std::sin(inclination) * std::sin(azimuth),
                                length * std::cos(inclination)) +
                     coordinates[0];

    Eigen::Vector3d u = coordinates[1];

    // Create the coordinates in the mesh
    std::transform(std::next(coordinates.begin(), 1),
                   std::prev(coordinates.end(), 1),
                   std::next(coordinates.begin(), 2),
                   [&](auto const& end_position) {

                       // Cross product the previous vector and a random vector
                       // which gives a vector that is perpendicular to the both
                       // vectors
                       Eigen::Vector3d k = u.cross(Coordinate::Random()).normalized();

                       // Rotate the perpendicular vector about the axis by
                       // a random angle to ensure uniform distribution
                       auto const random_angle = random_angle_gen(gen);

                       Eigen::Matrix3d R = rotationMatrix(random_angle, u.normalized());

                       k = R * k;

                       if (std::abs(k.dot(u)) > 1e-10)
                       {
                           std::cout << "Rotation axis not perpendicular to segment!\n";
                           throw 0;
                       }

                       R = rotationMatrix(bond_angle, k);

                       Eigen::Vector3d const v = R * u;

                       u = v;

                       return end_position + v;
                   });

    std::cout << "End to end distance : " << (coordinates.back() - coordinates.front()).norm()
              << " (numerical),\n"
              << "                      " << length * std::sqrt(2.0 * segments)
              << " (theoretical).\n";

    // Check for coordinates outside the box
    if (ranges::find_if(coordinates, [&](auto const& val) { return !box.isInBox(val); }) !=
        coordinates.end())
    {
        std::cout << "We went outside the bounding box...\n";
    }

    // Print out the results to paraview
    auto unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();

    auto vtk_points = vtkSmartPointer<vtkPoints>::New();
    for (auto const& xyz : coordinates)
    {
        double coords[3] = {xyz(0), xyz(1), xyz(2)};
        vtk_points->InsertNextPoint(coords);
    }

    vtk_points->Squeeze();
    unstructured_grid->SetPoints(vtk_points);

    auto vtk_segments = vtkSmartPointer<vtkIdList>::New();

    for (int i = 0; i < segments; i++)
    {
        vtk_segments->Allocate(2);

        vtk_segments->InsertNextId(i + 0);
        vtk_segments->InsertNextId(i + 1);

        unstructured_grid->InsertNextCell(VTK_LINE, vtk_segments);

        vtk_segments->Reset();
    }

    // Plot the end-to-end vector
    // vtk_segments->Allocate(2);
    // vtk_segments->InsertNextId(0);
    // vtk_segments->InsertNextId(segments);
    // unstructured_grid->InsertNextCell(VTK_LINE, vtk_segments);
    // vtk_segments->Reset();

    auto xml_writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();

    std::string vtk_filename = "PolymerChain.vtu";

    xml_writer->SetFileName(vtk_filename.c_str());
    xml_writer->SetInputData(unstructured_grid);
    xml_writer->SetDataModeToAscii();
    xml_writer->Write();

    return 0;
}
