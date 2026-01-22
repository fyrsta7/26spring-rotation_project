 *****************************************************************************/

#include "effects_image_provider.hpp"

#include <QPainter>
#include <QUrl>
#include <QUrlQuery>
#include <QPainterPath>

#include <memory>

#include "qt.hpp" // VLC_WEAK

// Qt private exported function
QT_BEGIN_NAMESPACE
extern void VLC_WEAK qt_blurImage(QImage &blurImage, qreal radius, bool quality, int transposed = 0);
QT_END_NAMESPACE


namespace {

class IEffect
{
public:
    virtual QImage generate(const QSize& size) const = 0;
    virtual ~IEffect() = default;
};

class RectDropShadowEffect : public IEffect
{

public:
    explicit RectDropShadowEffect(const QVariantMap& settings)
        : m_blurRadius(settings["blurRadius"].toReal())
        , m_color(settings["color"].value<QColor>())
        , m_xOffset(settings["xOffset"].toReal())
        , m_yOffset(settings["yOffset"].toReal())
    { }

    QImage generate(const QSize& size) const override
    {
        QImage mask(size, QImage::Format_ARGB32_Premultiplied);
        mask.fill(m_color);
        return generate(mask);
    }

    QImage generate(const QImage& mask) const
    {
        if (Q_UNLIKELY(!&qt_blurImage))
        {
            qWarning("qt_blurImage() is not available! Drop shadow will not work!");
            return {};
        }

        // Create a new image with boundaries containing the mask and effect.
        QImage ret(boundingSize(mask.size()), QImage::Format_ARGB32_Premultiplied);
        ret.fill(0);

        assert(!ret.isNull());
        {
            // Copy the mask
            QPainter painter(&ret);
            painter.setCompositionMode(QPainter::CompositionMode_Source);
            const auto radius = effectiveBlurRadius();
            painter.drawImage(radius + m_xOffset, radius + m_yOffset, mask);
        }

        // Blur the mask
        qt_blurImage(ret, effectiveBlurRadius(), false);

        return ret;
    }

    constexpr QSize boundingSize(const QSize& size) const
    {
        // Size of bounding rectangle of the effect
        const qreal radius = 2 * effectiveBlurRadius();
        return size + QSize(qAbs(m_xOffset) + radius, qAbs(m_yOffset) + radius);
    }

protected:
    qreal m_blurRadius = 1.0;
    QColor m_color {63, 63, 63, 180};
    qreal m_xOffset = 0.0;
    qreal m_yOffset = 0.0;

private:
    constexpr qreal effectiveBlurRadius() const
    {
        // Translated blur radius for the Qt blur algorithm
        return 2.5 * (m_blurRadius + 1);
    }
};

class RoundedRectDropShadowEffect : public RectDropShadowEffect
{
public:
    explicit RoundedRectDropShadowEffect(const QVariantMap& settings)
        : RectDropShadowEffect(settings)
        , m_xRadius(settings["xRadius"].toReal())
        , m_yRadius(settings["yRadius"].toReal())
    { }

    QImage generate(const QSize& size) const override
    {
        assert(!(qFuzzyIsNull(m_xRadius) && qFuzzyIsNull(m_yRadius))); // use RectDropShadowEffect instead

        QImage mask(size, QImage::Format_ARGB32_Premultiplied);
        mask.fill(Qt::transparent);

        assert(!mask.isNull());
        {
            QPainter painter(&mask);
            painter.setRenderHint(QPainter::Antialiasing);
            painter.setPen(m_color);
