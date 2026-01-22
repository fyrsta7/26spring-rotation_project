#include <LibWeb/Painting/BorderPainting.h>
#include <LibWeb/Painting/PaintContext.h>

namespace Web::Painting {

BorderRadiusData normalized_border_radius_data(Layout::Node const& node, Gfx::FloatRect const& rect, CSS::LengthPercentage top_left_radius, CSS::LengthPercentage top_right_radius, CSS::LengthPercentage bottom_right_radius, CSS::LengthPercentage bottom_left_radius)
{
    // FIXME: Some values should be relative to the height() if specified, but which?
    //        Spec just says "Refer to corresponding dimension of the border box."
    //        For now, all relative values are relative to the width.
    auto width_length = CSS::Length::make_px(rect.width());
    auto bottom_left_radius_px = bottom_left_radius.resolved(node, width_length).to_px(node);
    auto bottom_right_radius_px = bottom_right_radius.resolved(node, width_length).to_px(node);
    auto top_left_radius_px = top_left_radius.resolved(node, width_length).to_px(node);
    auto top_right_radius_px = top_right_radius.resolved(node, width_length).to_px(node);

    // Scale overlapping curves according to https://www.w3.org/TR/css-backgrounds-3/#corner-overlap
    auto f = 1.0f;
    auto width_reciprocal = 1.0f / rect.width();
    auto height_reciprocal = 1.0f / rect.height();
    f = max(f, width_reciprocal * (top_left_radius_px + top_right_radius_px));
    f = max(f, height_reciprocal * (top_right_radius_px + bottom_right_radius_px));
    f = max(f, width_reciprocal * (bottom_left_radius_px + bottom_right_radius_px));
    f = max(f, height_reciprocal * (top_left_radius_px + bottom_left_radius_px));

    f = 1.0f / f;

    top_left_radius_px = (int)(top_left_radius_px * f);
    top_right_radius_px = (int)(top_right_radius_px * f);
