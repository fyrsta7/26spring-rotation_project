// Check collision between two circles
bool CheckCollisionCircles(Vector2 center1, float radius1, Vector2 center2, float radius2)
{
    bool collision = false;

    float dx = center2.x - center1.x;      // X distance between centers
    float dy = center2.y - center1.y;      // Y distance between centers

    float distance = sqrtf(dx*dx + dy*dy); // Distance between centers

    if (distance <= (radius1 + radius2)) collision = true;

    return collision;
}